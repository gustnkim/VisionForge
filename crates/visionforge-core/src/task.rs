use std::path::Path;

use chrono::Utc;
use rusqlite::{OptionalExtension, params};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::project::{CoreError, initialize_schema, open_database, read_manifest};

const TASK_SPEC_SCHEMA_VERSION: u32 = 1;
const MAX_SCENARIO_LENGTH: usize = 4_000;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct OutputPolicy {
    pub present_folder: String,
    pub absent_folder: String,
    pub review_folder: String,
    pub failed_folder: String,
    pub positive_threshold: f64,
    pub negative_threshold: f64,
    pub copy_mode: String,
}

impl Default for OutputPolicy {
    fn default() -> Self {
        Self {
            present_folder: "대상_포함".to_owned(),
            absent_folder: "대상_미포함".to_owned(),
            review_folder: "검토_필요".to_owned(),
            failed_folder: "처리_실패".to_owned(),
            positive_threshold: 0.85,
            negative_threshold: 0.35,
            copy_mode: "copy_original".to_owned(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct GenerationPolicySpec {
    pub scale_min: f64,
    pub scale_max: f64,
    pub rotation_min: f64,
    pub rotation_max: f64,
    pub brightness_min: f64,
    pub brightness_max: f64,
    pub contrast_min: f64,
    pub contrast_max: f64,
    pub blur_radius_max: f64,
    pub noise_stddev_max: f64,
    pub occlusion_max: f64,
}

impl Default for GenerationPolicySpec {
    fn default() -> Self {
        Self {
            scale_min: 0.45,
            scale_max: 1.25,
            rotation_min: -25.0,
            rotation_max: 25.0,
            brightness_min: 0.78,
            brightness_max: 1.22,
            contrast_min: 0.9,
            contrast_max: 1.1,
            blur_radius_max: 0.7,
            noise_stddev_max: 0.0,
            occlusion_max: 0.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TaskSpecInput {
    pub task_type: String,
    pub scenario_description: String,
    pub output_policy: OutputPolicy,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct TaskSpec {
    pub schema_version: u32,
    pub id: String,
    pub revision: u64,
    pub task_type: String,
    pub pipeline_id: String,
    pub class_id: String,
    pub class_name: String,
    pub scenario_description: String,
    pub compiled_tags: Vec<String>,
    pub generation_policy: GenerationPolicySpec,
    pub output_policy: OutputPolicy,
    pub compiler: String,
    pub warnings: Vec<String>,
    pub created_at: String,
}

fn contains_any(value: &str, terms: &[&str]) -> bool {
    terms.iter().any(|term| value.contains(term))
}

fn push_tag(tags: &mut Vec<String>, value: &str) {
    if !tags.iter().any(|existing| existing == value) {
        tags.push(value.to_owned());
    }
}

fn compile_generation_policy(
    description: &str,
) -> (GenerationPolicySpec, Vec<String>, Vec<String>) {
    let normalized = description.to_lowercase();
    let mut policy = GenerationPolicySpec::default();
    let mut tags = Vec::new();
    let mut warnings = Vec::new();

    if contains_any(
        &normalized,
        &["멀리", "작게", "작은 대상", "far", "small target"],
    ) {
        policy.scale_min = 0.18;
        policy.scale_max = 0.85;
        push_tag(&mut tags, "small_target");
    }
    if contains_any(
        &normalized,
        &["가까이", "크게", "큰 대상", "close-up", "large target"],
    ) {
        policy.scale_max = 1.4;
        push_tag(&mut tags, "large_target");
    }
    if contains_any(&normalized, &["회전", "기울", "비스듬", "rotate", "tilt"]) {
        policy.rotation_min = -42.0;
        policy.rotation_max = 42.0;
        push_tag(&mut tags, "strong_rotation");
    }
    if contains_any(&normalized, &["어두", "야간", "밤", "dark", "night"]) {
        policy.brightness_min = 0.48;
        policy.brightness_max = 1.05;
        push_tag(&mut tags, "dark_lighting");
    }
    if contains_any(&normalized, &["밝은", "강한 조명", "bright", "overexposed"]) {
        policy.brightness_max = 1.48;
        push_tag(&mut tags, "bright_lighting");
    }
    if contains_any(&normalized, &["역광", "반사", "backlit", "reflection"]) {
        policy.brightness_min = policy.brightness_min.min(0.6);
        policy.brightness_max = policy.brightness_max.max(1.4);
        policy.contrast_min = 0.72;
        policy.contrast_max = 1.34;
        push_tag(&mut tags, "difficult_lighting");
    }
    if contains_any(
        &normalized,
        &["흐림", "블러", "초점", "blur", "out of focus"],
    ) {
        policy.blur_radius_max = 2.2;
        push_tag(&mut tags, "blur");
    }
    if contains_any(
        &normalized,
        &["노이즈", "저화질", "압축", "noise", "compression"],
    ) {
        policy.noise_stddev_max = 0.06;
        policy.blur_radius_max = policy.blur_radius_max.max(1.3);
        push_tag(&mut tags, "low_image_quality");
    }
    if contains_any(&normalized, &["가림", "가려", "occlusion", "occluded"]) {
        policy.occlusion_max = 0.42;
        push_tag(&mut tags, "occlusion");
    }
    if contains_any(&normalized, &["실내", "indoor"]) {
        push_tag(&mut tags, "indoor");
    }
    if contains_any(&normalized, &["실외", "야외", "도로", "outdoor"]) {
        push_tag(&mut tags, "outdoor");
    }
    if contains_any(
        &normalized,
        &["여러", "다수", "동시에", "multiple", "many objects"],
    ) {
        push_tag(&mut tags, "multiple_objects_requested");
        warnings.push(
            "다중 대상 합성은 현재 v1 생성기에서 지원하지 않아 대상 1개 합성으로 제한됩니다."
                .to_owned(),
        );
    }
    if contains_any(&normalized, &["원근", "광각", "perspective", "wide angle"]) {
        push_tag(&mut tags, "perspective_requested");
        warnings.push(
            "원근 변환은 현재 v1 생성기에서 지원하지 않아 크기·회전 변화로 대체됩니다.".to_owned(),
        );
    }
    if description.trim().is_empty() {
        warnings.push("상황 설명이 없어 기본 크기·회전·조명 분포를 사용합니다.".to_owned());
    }

    tags.sort();
    (policy, tags, warnings)
}

fn validate_folder_name(value: &str, label: &str) -> Result<String, CoreError> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(CoreError::InvalidProject(format!(
            "{label} 폴더 이름을 입력해 주세요."
        )));
    }
    if trimmed.chars().count() > 64 {
        return Err(CoreError::InvalidProject(format!(
            "{label} 폴더 이름은 64자 이하여야 합니다."
        )));
    }
    if trimmed == "."
        || trimmed == ".."
        || trimmed.ends_with(['.', ' '])
        || trimmed
            .chars()
            .any(|character| character.is_control() || r#"<>:"/\|?*"#.contains(character))
    {
        return Err(CoreError::InvalidProject(format!(
            "{label} 폴더 이름에 사용할 수 없는 문자가 있습니다."
        )));
    }
    let reserved = [
        "CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8",
        "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
    ];
    if reserved
        .iter()
        .any(|item| item.eq_ignore_ascii_case(trimmed))
    {
        return Err(CoreError::InvalidProject(format!(
            "{label} 폴더 이름은 운영체제 예약어입니다."
        )));
    }
    Ok(trimmed.to_owned())
}

pub(crate) fn validate_output_policy(policy: &OutputPolicy) -> Result<OutputPolicy, CoreError> {
    if !policy.positive_threshold.is_finite()
        || !policy.negative_threshold.is_finite()
        || !(0.0..=1.0).contains(&policy.positive_threshold)
        || !(0.0..=1.0).contains(&policy.negative_threshold)
        || policy.negative_threshold >= policy.positive_threshold
    {
        return Err(CoreError::InvalidProject(
            "음성 기준은 양성 기준보다 낮아야 하며 두 값은 0과 1 사이여야 합니다.".to_owned(),
        ));
    }
    if policy.copy_mode != "copy_original" {
        return Err(CoreError::InvalidProject(
            "현재 지원하는 결과 저장 방식은 copy_original뿐입니다.".to_owned(),
        ));
    }
    let present_folder = validate_folder_name(&policy.present_folder, "포함")?;
    let absent_folder = validate_folder_name(&policy.absent_folder, "미포함")?;
    let review_folder = validate_folder_name(&policy.review_folder, "검토 필요")?;
    let failed_folder = validate_folder_name(&policy.failed_folder, "처리 실패")?;
    let mut normalized = [
        present_folder.to_lowercase(),
        absent_folder.to_lowercase(),
        review_folder.to_lowercase(),
        failed_folder.to_lowercase(),
    ];
    normalized.sort();
    if normalized.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(CoreError::InvalidProject(
            "결과 상태별 폴더 이름은 서로 달라야 합니다.".to_owned(),
        ));
    }
    Ok(OutputPolicy {
        present_folder,
        absent_folder,
        review_folder,
        failed_folder,
        positive_threshold: policy.positive_threshold,
        negative_threshold: policy.negative_threshold,
        copy_mode: policy.copy_mode.clone(),
    })
}

fn build_task_spec(
    project_path: &Path,
    input: &TaskSpecInput,
    revision: u64,
) -> Result<TaskSpec, CoreError> {
    let project = read_manifest(project_path)?;
    if input.task_type != "object_presence" {
        return Err(CoreError::InvalidProject(format!(
            "현재 구현된 작업 유형은 object_presence뿐입니다: {}",
            input.task_type
        )));
    }
    let scenario = input.scenario_description.trim();
    if scenario.chars().count() > MAX_SCENARIO_LENGTH {
        return Err(CoreError::InvalidProject(format!(
            "상황 설명은 {MAX_SCENARIO_LENGTH}자 이하여야 합니다."
        )));
    }
    let output_policy = validate_output_policy(&input.output_policy)?;
    let (generation_policy, compiled_tags, warnings) = compile_generation_policy(scenario);
    Ok(TaskSpec {
        schema_version: TASK_SPEC_SCHEMA_VERSION,
        id: Uuid::new_v4().to_string(),
        revision,
        task_type: input.task_type.clone(),
        pipeline_id: "single_class_object_detection".to_owned(),
        class_id: project.class_id,
        class_name: project.class_name,
        scenario_description: scenario.to_owned(),
        compiled_tags,
        generation_policy,
        output_policy,
        compiler: "visionforge-keyword-compiler-v1".to_owned(),
        warnings,
        created_at: Utc::now().to_rfc3339(),
    })
}

pub fn default_task_spec(project_path: impl AsRef<Path>) -> Result<TaskSpec, CoreError> {
    build_task_spec(
        project_path.as_ref(),
        &TaskSpecInput {
            task_type: "object_presence".to_owned(),
            scenario_description: String::new(),
            output_policy: OutputPolicy::default(),
        },
        0,
    )
}

pub fn current_task_spec(project_path: impl AsRef<Path>) -> Result<TaskSpec, CoreError> {
    let project_path = project_path.as_ref();
    let _project = read_manifest(project_path)?;
    let connection = open_database(project_path)?;
    initialize_schema(&connection)?;
    let serialized: Option<String> = connection
        .query_row(
            "SELECT spec_json FROM task_specs ORDER BY revision DESC LIMIT 1",
            [],
            |row| row.get(0),
        )
        .optional()?;
    if let Some(value) = serialized {
        return Ok(serde_json::from_str(&value)?);
    }
    drop(connection);
    save_task_spec(
        project_path,
        &TaskSpecInput {
            task_type: "object_presence".to_owned(),
            scenario_description: String::new(),
            output_policy: OutputPolicy::default(),
        },
    )
}

pub fn save_task_spec(
    project_path: impl AsRef<Path>,
    input: &TaskSpecInput,
) -> Result<TaskSpec, CoreError> {
    let project_path = project_path.as_ref();
    let mut connection = open_database(project_path)?;
    initialize_schema(&connection)?;
    let previous: Option<i64> =
        connection.query_row("SELECT MAX(revision) FROM task_specs", [], |row| row.get(0))?;
    let revision = u64::try_from(previous.unwrap_or(0).saturating_add(1))
        .map_err(|_| CoreError::InvalidProject("작업 명세 리비전이 너무 큽니다.".to_owned()))?;
    let spec = build_task_spec(project_path, input, revision)?;
    let transaction = connection.transaction()?;
    transaction.execute(
        "INSERT INTO task_specs(
            id, revision, task_type, pipeline_id, spec_json, created_at
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        params![
            spec.id,
            i64::try_from(spec.revision).map_err(|_| CoreError::InvalidProject(
                "작업 명세 리비전이 너무 큽니다.".to_owned()
            ))?,
            spec.task_type,
            spec.pipeline_id,
            serde_json::to_string(&spec)?,
            spec.created_at,
        ],
    )?;
    transaction.commit()?;
    Ok(spec)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::create_project;

    #[test]
    fn scenario_compiler_changes_generation_policy() {
        let root = tempfile::tempdir().expect("temporary directory");
        let project = create_project(root.path(), "동적 작업", "표지판").expect("project");
        let spec = save_task_spec(
            &project.path,
            &TaskSpecInput {
                task_type: "object_presence".to_owned(),
                scenario_description: "야외에서 멀리 작게 보이고 일부 가려진 저화질 사진"
                    .to_owned(),
                output_policy: OutputPolicy::default(),
            },
        )
        .expect("task spec");

        assert_eq!(spec.revision, 1);
        assert!(spec.compiled_tags.contains(&"small_target".to_owned()));
        assert!(spec.compiled_tags.contains(&"occlusion".to_owned()));
        assert!(spec.generation_policy.scale_min < 0.2);
        assert!(spec.generation_policy.occlusion_max > 0.4);
        assert!(spec.generation_policy.noise_stddev_max > 0.0);
        assert_eq!(current_task_spec(&project.path).expect("load"), spec);
    }

    #[test]
    fn unsafe_output_policy_is_rejected() {
        let root = tempfile::tempdir().expect("temporary directory");
        let project = create_project(root.path(), "안전", "상자").expect("project");
        let output = OutputPolicy {
            present_folder: "../outside".to_owned(),
            ..OutputPolicy::default()
        };
        let error = save_task_spec(
            &project.path,
            &TaskSpecInput {
                task_type: "object_presence".to_owned(),
                scenario_description: String::new(),
                output_policy: output,
            },
        )
        .expect_err("unsafe path must fail");
        assert!(error.to_string().contains("사용할 수 없는 문자"));
    }
}
