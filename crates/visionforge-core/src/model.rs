use std::fs::{self, File};
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

use chrono::Utc;
use rusqlite::{OptionalExtension, params};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::project::{
    BoundingBoxInput, CoreError, JobSummary, WarningInput, initialize_schema, open_database,
    read_manifest,
};
use crate::task::TaskSpec;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all(serialize = "camelCase", deserialize = "snake_case"))]
pub struct TrainingMetricsInput {
    pub evaluation_split: String,
    pub positive_images: i64,
    pub negative_images: i64,
    pub true_positives: i64,
    pub false_positives: i64,
    pub false_negatives: i64,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    pub mean_iou: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingResultInput {
    pub status: String,
    pub model_path: String,
    pub metrics_path: String,
    pub model_id: Option<String>,
    pub checksum_sha256: Option<String>,
    pub engine_name: Option<String>,
    #[serde(default)]
    pub deployment_status: Option<String>,
    pub metrics: Option<TrainingMetricsInput>,
    #[serde(default)]
    pub warnings: Vec<WarningInput>,
    pub error_code: Option<String>,
    pub error_message: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all(serialize = "camelCase", deserialize = "snake_case"))]
pub struct ModelPackageResultInput {
    pub status: String,
    pub package_path: String,
    pub package_id: Option<String>,
    pub package_checksum_sha256: Option<String>,
    pub class_id: Option<String>,
    pub class_name: Option<String>,
    pub engine_name: Option<String>,
    #[serde(default)]
    pub deployment_status: Option<String>,
    pub model_path: Option<String>,
    pub metrics_path: Option<String>,
    #[serde(default)]
    pub task_spec_path: Option<String>,
    #[serde(default)]
    pub manifest: serde_json::Value,
    #[serde(default)]
    pub warnings: Vec<WarningInput>,
    pub error_code: Option<String>,
    pub error_message: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelVersionSummary {
    pub id: String,
    pub dataset_id: String,
    pub status: String,
    pub deployment_status: String,
    pub engine_name: String,
    pub class_id: String,
    pub class_name: String,
    pub origin: String,
    pub model_path: String,
    pub checksum_sha256: String,
    pub metrics: TrainingMetricsInput,
    pub warnings: Vec<WarningInput>,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all(serialize = "camelCase", deserialize = "snake_case"))]
pub struct DetectionInput {
    pub class_id: String,
    pub class_name: String,
    pub confidence: f64,
    pub bounding_box: BoundingBoxInput,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceImageInput {
    pub status: String,
    pub input_path: String,
    pub output_path: String,
    #[serde(default)]
    pub detections: Vec<DetectionInput>,
    #[serde(default)]
    pub max_confidence: Option<f64>,
    pub checksum_sha256: Option<String>,
    pub width: Option<i64>,
    pub height: Option<i64>,
    pub elapsed_ms: Option<f64>,
    pub error_code: Option<String>,
    pub error_message: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InferenceItemRecord {
    pub id: String,
    pub status: String,
    pub input_path: String,
    pub output_path: Option<String>,
    pub detections: Vec<DetectionInput>,
    pub max_confidence: Option<f64>,
    pub decision: String,
    pub routed_path: Option<String>,
    pub routing_error: Option<String>,
    pub elapsed_ms: Option<f64>,
    pub error_code: Option<String>,
    pub error_message: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RoutingSummary {
    pub root_path: String,
    pub manifest_path: String,
    pub present_count: u64,
    pub absent_count: u64,
    pub review_count: u64,
    pub failed_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InferenceBatchResult {
    pub run_id: String,
    pub model_id: String,
    pub job: JobSummary,
    pub items: Vec<InferenceItemRecord>,
    pub routing: Option<RoutingSummary>,
}

fn sha256_file(path: &Path) -> Result<String, CoreError> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn project_relative(project_path: &Path, value: &Path) -> Result<(PathBuf, String), CoreError> {
    let project = fs::canonicalize(project_path)?;
    let canonical = fs::canonicalize(value)?;
    let relative = canonical.strip_prefix(&project).map_err(|_| {
        CoreError::InvalidProject("결과 파일이 프로젝트 저장소 밖에 있습니다.".to_owned())
    })?;
    let relative = relative.to_string_lossy().replace('\\', "/");
    Ok((canonical, relative))
}

fn read_job(connection: &rusqlite::Connection, job_id: &str) -> Result<JobSummary, CoreError> {
    connection
        .query_row(
            "SELECT id, job_type, status, total_items, completed_items, failed_items,
                    created_at, updated_at
             FROM jobs WHERE id = ?1",
            [job_id],
            |row| {
                let total: i64 = row.get(3)?;
                let completed: i64 = row.get(4)?;
                let failed: i64 = row.get(5)?;
                Ok(JobSummary {
                    id: row.get(0)?,
                    job_type: row.get(1)?,
                    status: row.get(2)?,
                    total_items: total.max(0) as u64,
                    completed_items: completed.max(0) as u64,
                    failed_items: failed.max(0) as u64,
                    created_at: row.get(6)?,
                    updated_at: row.get(7)?,
                })
            },
        )
        .map_err(CoreError::from)
}

pub fn record_training_result(
    project_path: impl AsRef<Path>,
    job_id: &str,
    dataset_id: &str,
    result: &TrainingResultInput,
) -> Result<ModelVersionSummary, CoreError> {
    if result.status != "succeeded" {
        return Err(CoreError::InvalidProject(
            result
                .error_message
                .clone()
                .unwrap_or_else(|| "학습 결과가 실패 상태입니다.".to_owned()),
        ));
    }
    let project_path = project_path.as_ref();
    let manifest = read_manifest(project_path)?;
    let (model_path, relative_model) =
        project_relative(project_path, Path::new(&result.model_path))?;
    let (_, relative_metrics) = project_relative(project_path, Path::new(&result.metrics_path))?;
    let checksum = result
        .checksum_sha256
        .as_deref()
        .ok_or_else(|| CoreError::InvalidProject("모델 체크섬이 누락됐습니다.".to_owned()))?;
    if sha256_file(&model_path)? != checksum {
        return Err(CoreError::InvalidProject(
            "학습 모델 체크섬이 일치하지 않습니다.".to_owned(),
        ));
    }
    let model_id = result
        .model_id
        .clone()
        .ok_or_else(|| CoreError::InvalidProject("모델 ID가 누락됐습니다.".to_owned()))?;
    let engine_name = result
        .engine_name
        .clone()
        .ok_or_else(|| CoreError::InvalidProject("학습 엔진 이름이 누락됐습니다.".to_owned()))?;
    let deployment_status = result
        .deployment_status
        .as_deref()
        .unwrap_or("experimental");
    if !matches!(
        deployment_status,
        "experimental" | "candidate" | "qualified"
    ) {
        return Err(CoreError::InvalidProject(
            "모델 배포 상태가 올바르지 않습니다.".to_owned(),
        ));
    }
    let metrics = result
        .metrics
        .clone()
        .ok_or_else(|| CoreError::InvalidProject("학습 지표가 누락됐습니다.".to_owned()))?;
    let created_at = Utc::now().to_rfc3339();

    let mut connection = open_database(project_path)?;
    initialize_schema(&connection)?;
    let dataset_exists = connection
        .query_row(
            "SELECT 1 FROM dataset_versions WHERE id = ?1 AND status = 'ready'",
            [dataset_id],
            |_| Ok(()),
        )
        .optional()?
        .is_some();
    if !dataset_exists {
        return Err(CoreError::InvalidProject(
            "학습 데이터셋 버전을 찾을 수 없습니다.".to_owned(),
        ));
    }
    let transaction = connection.transaction()?;
    transaction.execute(
        "INSERT INTO model_versions(
            id, dataset_id, status, deployment_status, engine_name, class_id, class_name, origin,
            model_path, model_checksum_sha256, metrics_path, metrics_json,
            warnings_json, created_at
         ) VALUES (?1, ?2, 'ready', ?3, ?4, ?5, ?6, 'trained', ?7, ?8, ?9, ?10, ?11, ?12)",
        params![
            model_id,
            dataset_id,
            deployment_status,
            engine_name,
            manifest.class_id,
            manifest.class_name,
            relative_model,
            checksum,
            relative_metrics,
            serde_json::to_string(&metrics)?,
            serde_json::to_string(&result.warnings)?,
            created_at,
        ],
    )?;
    transaction.execute(
        "INSERT INTO job_items(id, job_id, status, result_json, updated_at)
         VALUES (?1, ?2, 'succeeded', ?3, ?4)",
        params![
            Uuid::new_v4().to_string(),
            job_id,
            serde_json::to_string(result)?,
            created_at,
        ],
    )?;
    transaction.execute(
        "UPDATE jobs SET status = 'succeeded', completed_items = 1,
                         failed_items = 0, updated_at = ?2 WHERE id = ?1",
        params![job_id, created_at],
    )?;
    transaction.commit()?;

    Ok(ModelVersionSummary {
        id: model_id,
        dataset_id: dataset_id.to_owned(),
        status: "ready".to_owned(),
        deployment_status: deployment_status.to_owned(),
        engine_name,
        class_id: manifest.class_id,
        class_name: manifest.class_name,
        origin: "trained".to_owned(),
        model_path: model_path.to_string_lossy().into_owned(),
        checksum_sha256: checksum.to_owned(),
        metrics,
        warnings: result.warnings.clone(),
        created_at,
    })
}

pub fn model_file_path(
    project_path: impl AsRef<Path>,
    model_id: &str,
) -> Result<PathBuf, CoreError> {
    let project_path = project_path.as_ref();
    let _manifest = read_manifest(project_path)?;
    let connection = open_database(project_path)?;
    initialize_schema(&connection)?;
    let relative: String = connection.query_row(
        "SELECT model_path FROM model_versions WHERE id = ?1 AND status = 'ready'",
        [model_id],
        |row| row.get(0),
    )?;
    let (path, _) = project_relative(project_path, &project_path.join(relative))?;
    Ok(path)
}

pub fn model_package_sources(
    project_path: impl AsRef<Path>,
    model_id: &str,
) -> Result<(PathBuf, PathBuf), CoreError> {
    let project_path = project_path.as_ref();
    let _manifest = read_manifest(project_path)?;
    let connection = open_database(project_path)?;
    initialize_schema(&connection)?;
    let (model_relative, metrics_relative): (String, String) = connection.query_row(
        "SELECT model_path, metrics_path FROM model_versions
         WHERE id = ?1 AND status = 'ready'",
        [model_id],
        |row| Ok((row.get(0)?, row.get(1)?)),
    )?;
    let (model_path, _) = project_relative(project_path, &project_path.join(model_relative))?;
    let (metrics_path, _) = project_relative(project_path, &project_path.join(metrics_relative))?;
    Ok((model_path, metrics_path))
}

pub fn latest_model_version(
    project_path: impl AsRef<Path>,
) -> Result<Option<ModelVersionSummary>, CoreError> {
    type LatestModelRow = (
        String,
        String,
        String,
        String,
        String,
        String,
        String,
        String,
        String,
        String,
        String,
        String,
    );

    let project_path = project_path.as_ref();
    let _manifest = read_manifest(project_path)?;
    let connection = open_database(project_path)?;
    initialize_schema(&connection)?;
    let row: Option<LatestModelRow> = connection
        .query_row(
            "SELECT id, dataset_id, deployment_status, engine_name, class_id, class_name, origin,
                    model_path, model_checksum_sha256, metrics_json, warnings_json,
                    created_at
             FROM model_versions WHERE status = 'ready'
             ORDER BY created_at DESC, id DESC LIMIT 1",
            [],
            |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                    row.get(5)?,
                    row.get(6)?,
                    row.get(7)?,
                    row.get(8)?,
                    row.get(9)?,
                    row.get(10)?,
                    row.get(11)?,
                ))
            },
        )
        .optional()?;
    let Some((
        id,
        dataset_id,
        deployment_status,
        engine_name,
        class_id,
        class_name,
        origin,
        model_relative,
        checksum_sha256,
        metrics_json,
        warnings_json,
        created_at,
    )) = row
    else {
        return Ok(None);
    };
    let (model_path, _) = project_relative(project_path, &project_path.join(model_relative))?;
    Ok(Some(ModelVersionSummary {
        id,
        dataset_id,
        status: "ready".to_owned(),
        deployment_status,
        engine_name,
        class_id,
        class_name,
        origin,
        model_path: model_path.to_string_lossy().into_owned(),
        checksum_sha256,
        metrics: serde_json::from_str(&metrics_json)?,
        warnings: serde_json::from_str(&warnings_json)?,
        created_at,
    }))
}

pub fn record_imported_model(
    project_path: impl AsRef<Path>,
    package: &ModelPackageResultInput,
) -> Result<ModelVersionSummary, CoreError> {
    if package.status != "succeeded" {
        return Err(CoreError::InvalidProject(
            package
                .error_message
                .clone()
                .unwrap_or_else(|| "모델 패키지 검증에 실패했습니다.".to_owned()),
        ));
    }
    let project_path = project_path.as_ref();
    let _project = read_manifest(project_path)?;
    let package_id = package
        .package_id
        .as_deref()
        .ok_or_else(|| CoreError::InvalidProject("패키지 ID가 누락됐습니다.".to_owned()))?;
    let class_id = package
        .class_id
        .clone()
        .ok_or_else(|| CoreError::InvalidProject("패키지 클래스 ID가 누락됐습니다.".to_owned()))?;
    let class_name = package.class_name.clone().ok_or_else(|| {
        CoreError::InvalidProject("패키지 클래스 이름이 누락됐습니다.".to_owned())
    })?;
    let engine_name = package
        .engine_name
        .clone()
        .ok_or_else(|| CoreError::InvalidProject("패키지 엔진 이름이 누락됐습니다.".to_owned()))?;
    let deployment_status = package
        .deployment_status
        .as_deref()
        .unwrap_or("experimental");
    if !matches!(
        deployment_status,
        "experimental" | "candidate" | "qualified"
    ) {
        return Err(CoreError::InvalidProject(
            "패키지 모델 배포 상태가 올바르지 않습니다.".to_owned(),
        ));
    }
    let model_source = package
        .model_path
        .as_deref()
        .ok_or_else(|| CoreError::InvalidProject("패키지 모델 파일이 누락됐습니다.".to_owned()))?;
    let metrics_source = package
        .metrics_path
        .as_deref()
        .ok_or_else(|| CoreError::InvalidProject("패키지 지표 파일이 누락됐습니다.".to_owned()))?;
    let (model_path, model_relative) = project_relative(project_path, Path::new(model_source))?;
    let (metrics_path, metrics_relative) =
        project_relative(project_path, Path::new(metrics_source))?;
    let (package_path, package_relative) =
        project_relative(project_path, Path::new(&package.package_path))?;
    let package_checksum = package
        .package_checksum_sha256
        .as_deref()
        .ok_or_else(|| CoreError::InvalidProject("패키지 체크섬이 누락됐습니다.".to_owned()))?;
    if sha256_file(&package_path)? != package_checksum {
        return Err(CoreError::InvalidProject(
            "모델 패키지 체크섬이 일치하지 않습니다.".to_owned(),
        ));
    }
    let metrics: TrainingMetricsInput = serde_json::from_slice(&fs::read(&metrics_path)?)?;
    let model_checksum = sha256_file(&model_path)?;
    let import_root = model_path.parent().and_then(Path::parent).ok_or_else(|| {
        CoreError::InvalidProject("가져온 모델 경로가 올바르지 않습니다.".to_owned())
    })?;
    let manifest_path = import_root.join("manifest.json");
    let (_, manifest_relative) = project_relative(project_path, &manifest_path)?;
    let manifest_checksum = sha256_file(&manifest_path)?;
    let created_at = Utc::now().to_rfc3339();
    let model_id = Uuid::new_v4().to_string();
    let dataset_id = format!("package-{package_id}");

    let mut connection = open_database(project_path)?;
    initialize_schema(&connection)?;
    let previous_version: Option<i64> = connection.query_row(
        "SELECT MAX(version_number) FROM dataset_versions",
        [],
        |row| row.get(0),
    )?;
    let version = previous_version.unwrap_or(0) + 1;
    let transaction = connection.transaction()?;
    transaction.execute(
        "INSERT INTO dataset_versions(
            id, version_number, status, manifest_path, checksum_sha256, seed,
            total_items, train_items, validation_items, test_items, created_at
         ) VALUES (?1, ?2, 'external_reference', ?3, ?4, 0, 0, 0, 0, 0, ?5)",
        params![
            dataset_id,
            version,
            manifest_relative,
            manifest_checksum,
            created_at
        ],
    )?;
    transaction.execute(
        "INSERT INTO model_versions(
            id, dataset_id, status, deployment_status, engine_name, class_id, class_name, origin,
            package_path, model_path, model_checksum_sha256, metrics_path,
            metrics_json, warnings_json, created_at
         ) VALUES (?1, ?2, 'ready', ?3, ?4, ?5, ?6, 'imported', ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
        params![
            model_id,
            dataset_id,
            deployment_status,
            engine_name,
            class_id,
            class_name,
            package_relative,
            model_relative,
            model_checksum,
            metrics_relative,
            serde_json::to_string(&metrics)?,
            serde_json::to_string(&package.warnings)?,
            created_at,
        ],
    )?;
    transaction.commit()?;

    Ok(ModelVersionSummary {
        id: model_id,
        dataset_id,
        status: "ready".to_owned(),
        deployment_status: deployment_status.to_owned(),
        engine_name,
        class_id,
        class_name,
        origin: "imported".to_owned(),
        model_path: model_path.to_string_lossy().into_owned(),
        checksum_sha256: model_checksum,
        metrics,
        warnings: package.warnings.clone(),
        created_at,
    })
}

pub fn record_inference_results(
    project_path: impl AsRef<Path>,
    job_id: &str,
    model_id: &str,
    confidence_threshold: f64,
    task_spec: &TaskSpec,
    results: &[InferenceImageInput],
) -> Result<InferenceBatchResult, CoreError> {
    let project_path = project_path.as_ref();
    let _manifest = read_manifest(project_path)?;
    let mut connection = open_database(project_path)?;
    initialize_schema(&connection)?;
    let transaction = connection.transaction()?;
    let (expected_class_id, expected_class_name): (String, String) = transaction.query_row(
        "SELECT class_id, class_name FROM model_versions WHERE id = ?1 AND status = 'ready'",
        [model_id],
        |row| Ok((row.get(0)?, row.get(1)?)),
    )?;
    if task_spec.class_id != expected_class_id && task_spec.class_name != expected_class_name {
        return Err(CoreError::InvalidProject(
            "작업 명세 클래스와 모델 클래스가 일치하지 않습니다.".to_owned(),
        ));
    }
    let run_id = Uuid::new_v4().to_string();
    let now = Utc::now().to_rfc3339();
    transaction.execute(
        "INSERT INTO inference_runs(
            id, job_id, model_id, confidence_threshold, task_spec_id,
            task_spec_revision, output_policy_json, created_at
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        params![
            run_id,
            job_id,
            model_id,
            confidence_threshold,
            task_spec.id,
            i64::try_from(task_spec.revision).map_err(|_| CoreError::InvalidProject(
                "작업 명세 리비전이 너무 큽니다.".to_owned()
            ))?,
            serde_json::to_string(&task_spec.output_policy)?,
            now
        ],
    )?;

    let mut completed = 0_i64;
    let mut failed = 0_i64;
    let mut records = Vec::with_capacity(results.len());
    for result in results {
        let input_asset: Option<(String, String)> = if let Ok((_, input_relative)) =
            project_relative(project_path, Path::new(&result.input_path))
        {
            transaction
                .query_row(
                    "SELECT id, original_name FROM image_assets
                     WHERE internal_path = ?1 AND role = 'inference_input' LIMIT 1",
                    [&input_relative],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )
                .optional()?
        } else {
            transaction
                .query_row(
                    "SELECT id, original_name FROM image_assets
                     WHERE original_path = ?1 AND role = 'inference_input'
                     ORDER BY created_at DESC LIMIT 1",
                    [&result.input_path],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )
                .optional()?
        };
        let (input_asset_id, input_name) = input_asset.ok_or_else(|| {
            CoreError::InvalidProject("추론 입력 자산을 찾을 수 없습니다.".to_owned())
        })?;
        let result_id = Uuid::new_v4().to_string();
        let mut output_asset_id = None;
        let mut stored_output_path = None;

        if result.status == "succeeded" {
            let max_confidence = result.max_confidence.ok_or_else(|| {
                CoreError::InvalidProject("추론 최고 후보 점수가 누락됐습니다.".to_owned())
            })?;
            if !max_confidence.is_finite() || !(0.0..=1.0).contains(&max_confidence) {
                return Err(CoreError::InvalidProject(
                    "추론 최고 후보 점수가 올바르지 않습니다.".to_owned(),
                ));
            }
            let (output_path, output_relative) =
                project_relative(project_path, Path::new(&result.output_path))?;
            let checksum = result.checksum_sha256.as_deref().ok_or_else(|| {
                CoreError::InvalidProject("추론 결과 체크섬이 누락됐습니다.".to_owned())
            })?;
            if sha256_file(&output_path)? != checksum {
                return Err(CoreError::InvalidProject(
                    "추론 결과 체크섬이 일치하지 않습니다.".to_owned(),
                ));
            }
            let width = result.width.ok_or_else(|| {
                CoreError::InvalidProject("추론 결과 너비가 누락됐습니다.".to_owned())
            })?;
            let height = result.height.ok_or_else(|| {
                CoreError::InvalidProject("추론 결과 높이가 누락됐습니다.".to_owned())
            })?;
            for detection in &result.detections {
                let box_value = &detection.bounding_box;
                if detection.class_id != expected_class_id
                    || !(0.0..=1.0).contains(&detection.confidence)
                    || box_value.x_min < 0
                    || box_value.y_min < 0
                    || box_value.x_max <= box_value.x_min
                    || box_value.y_max <= box_value.y_min
                    || box_value.x_max > width
                    || box_value.y_max > height
                {
                    return Err(CoreError::InvalidProject(
                        "추론 Detection 값이 올바르지 않습니다.".to_owned(),
                    ));
                }
            }
            let asset_id = Uuid::new_v4().to_string();
            let file_size = i64::try_from(fs::metadata(&output_path)?.len())
                .map_err(|_| CoreError::InvalidProject("추론 결과가 너무 큽니다.".to_owned()))?;
            let output_name = output_path
                .file_name()
                .and_then(|value| value.to_str())
                .unwrap_or("result.png");
            transaction.execute(
                "INSERT INTO image_assets(
                    id, role, original_path, original_name, internal_path, status,
                    checksum_sha256, image_format, width, height, file_size, has_alpha,
                    warnings_json, review_status, created_at
                 ) VALUES (?1, 'inference_result', ?2, ?3, ?4, 'succeeded',
                           ?5, 'PNG', ?6, ?7, ?8, 0, '[]', 'unreviewed', ?9)",
                params![
                    asset_id,
                    result.input_path,
                    output_name,
                    output_relative,
                    checksum,
                    width,
                    height,
                    file_size,
                    now,
                ],
            )?;
            output_asset_id = Some(asset_id);
            stored_output_path = Some(output_path.to_string_lossy().into_owned());
            completed += 1;
        } else {
            failed += 1;
        }

        transaction.execute(
            "INSERT INTO inference_results(
                id, run_id, input_asset_id, output_asset_id, status, max_confidence,
                decision, routed_path, routing_error, elapsed_ms,
                error_code, error_message, created_at
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, 'unrouted', NULL, NULL, ?7, ?8, ?9, ?10)",
            params![
                result_id,
                run_id,
                input_asset_id,
                output_asset_id,
                result.status,
                result.max_confidence,
                result.elapsed_ms,
                result.error_code,
                result.error_message,
                now,
            ],
        )?;
        for detection in &result.detections {
            let box_value = &detection.bounding_box;
            transaction.execute(
                "INSERT INTO detections(
                    id, inference_result_id, class_id, class_name, confidence,
                    x_min, y_min, x_max, y_max
                 ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                params![
                    Uuid::new_v4().to_string(),
                    result_id,
                    detection.class_id,
                    detection.class_name,
                    detection.confidence,
                    box_value.x_min,
                    box_value.y_min,
                    box_value.x_max,
                    box_value.y_max,
                ],
            )?;
        }
        transaction.execute(
            "INSERT INTO job_items(
                id, job_id, source_asset_id, status, error_code, error_message,
                result_json, updated_at
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                Uuid::new_v4().to_string(),
                job_id,
                input_asset_id,
                result.status,
                result.error_code,
                result.error_message,
                serde_json::to_string(result)?,
                now,
            ],
        )?;
        records.push(InferenceItemRecord {
            id: result_id,
            status: result.status.clone(),
            input_path: input_name,
            output_path: stored_output_path,
            detections: result.detections.clone(),
            max_confidence: result.max_confidence,
            decision: "unrouted".to_owned(),
            routed_path: None,
            routing_error: None,
            elapsed_ms: result.elapsed_ms,
            error_code: result.error_code.clone(),
            error_message: result.error_message.clone(),
        });
    }

    let status = if completed == 0 {
        "failed"
    } else if failed > 0 {
        "partial_failed"
    } else {
        "succeeded"
    };
    transaction.execute(
        "UPDATE jobs SET status = ?2, completed_items = ?3, failed_items = ?4,
                         updated_at = ?5 WHERE id = ?1",
        params![job_id, status, completed, failed, now],
    )?;
    transaction.commit()?;
    let job = read_job(&connection, job_id)?;
    Ok(InferenceBatchResult {
        run_id,
        model_id: model_id.to_owned(),
        job,
        items: records,
        routing: None,
    })
}
