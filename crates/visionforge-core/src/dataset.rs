use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufReader, Read};
use std::path::{Component, Path, PathBuf};

use chrono::Utc;
use rusqlite::{OptionalExtension, params};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::project::{CoreError, initialize_schema, open_database, read_manifest};
use crate::task::{GenerationPolicySpec, current_task_spec};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReviewUpdate {
    pub requested: u64,
    pub updated: u64,
    pub review_status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DatasetAnnotation {
    pub class_id: String,
    pub x_min: i64,
    pub y_min: i64,
    pub x_max: i64,
    pub y_max: i64,
    pub source: String,
    pub user_modified: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DatasetItem {
    pub asset_id: String,
    pub role: String,
    pub path: String,
    pub checksum_sha256: String,
    pub width: u32,
    pub height: u32,
    pub split: String,
    pub group_key: String,
    pub annotations: Vec<DatasetAnnotation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DatasetStats {
    pub total_items: u64,
    pub positive_items: u64,
    pub negative_items: u64,
    pub train_items: u64,
    pub validation_items: u64,
    pub test_items: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DatasetManifest {
    pub schema_version: u32,
    pub id: String,
    pub version: u64,
    pub project_id: String,
    pub class_id: String,
    pub class_name: String,
    #[serde(default)]
    pub task_spec_id: String,
    #[serde(default)]
    pub task_spec_revision: u64,
    #[serde(default)]
    pub generation_policy: GenerationPolicySpec,
    pub created_at: String,
    pub seed: i64,
    pub immutable: bool,
    pub stats: DatasetStats,
    pub warnings: Vec<String>,
    pub items: Vec<DatasetItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DatasetVersionSummary {
    pub id: String,
    pub version: u64,
    pub manifest_path: String,
    pub checksum_sha256: String,
    pub created_at: String,
    pub task_spec_id: String,
    pub task_spec_revision: u64,
    pub stats: DatasetStats,
    pub warnings: Vec<String>,
}

#[derive(Debug)]
struct Candidate {
    asset_id: String,
    role: String,
    internal_path: String,
    checksum_sha256: String,
    width: u32,
    height: u32,
    group_key: String,
    annotations: Vec<DatasetAnnotation>,
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

fn safe_project_asset(project_path: &Path, relative: &str) -> Result<PathBuf, CoreError> {
    let relative_path = Path::new(relative);
    if relative_path.is_absolute()
        || relative_path
            .components()
            .any(|component| matches!(component, Component::ParentDir | Component::Prefix(_)))
    {
        return Err(CoreError::InvalidProject(format!(
            "프로젝트 밖을 가리키는 자산 경로입니다: {relative}"
        )));
    }
    let full_path = project_path.join(relative_path);
    let canonical = fs::canonicalize(&full_path)?;
    let project_canonical = fs::canonicalize(project_path)?;
    if !canonical.starts_with(project_canonical) {
        return Err(CoreError::InvalidProject(format!(
            "프로젝트 밖을 가리키는 자산 경로입니다: {relative}"
        )));
    }
    Ok(canonical)
}

fn group_order(seed: i64, group: &str) -> [u8; 32] {
    Sha256::digest(format!("{seed}:{group}")).into()
}

fn split_groups(groups: impl Iterator<Item = String>, seed: i64) -> HashMap<String, String> {
    let mut groups = groups.collect::<Vec<_>>();
    groups.sort_by_key(|group| group_order(seed, group));
    groups.dedup();

    let count = groups.len();
    let (train_count, validation_count) = match count {
        0 => (0, 0),
        1 => (1, 0),
        2 => (1, 1),
        _ => {
            let validation = (count / 10).max(1);
            let test = (count / 10).max(1);
            (count - validation - test, validation)
        }
    };

    groups
        .into_iter()
        .enumerate()
        .map(|(index, group)| {
            let split = if index < train_count {
                "train"
            } else if index < train_count + validation_count {
                "validation"
            } else {
                "test"
            };
            (group, split.to_owned())
        })
        .collect()
}

pub fn set_image_review_status(
    project_path: impl AsRef<Path>,
    asset_ids: &[String],
    review_status: &str,
) -> Result<ReviewUpdate, CoreError> {
    if !matches!(
        review_status,
        "approved" | "excluded" | "unreviewed" | "needs_review"
    ) {
        return Err(CoreError::InvalidProject(format!(
            "지원하지 않는 검토 상태입니다: {review_status}"
        )));
    }
    let project_path = project_path.as_ref();
    let _manifest = read_manifest(project_path)?;
    let mut connection = open_database(project_path)?;
    initialize_schema(&connection)?;
    let transaction = connection.transaction()?;
    let mut updated = 0_u64;
    for asset_id in asset_ids {
        updated += transaction.execute(
            "UPDATE image_assets
             SET review_status = ?2
             WHERE id = ?1 AND status = 'succeeded'",
            params![asset_id, review_status],
        )? as u64;
    }
    transaction.commit()?;
    Ok(ReviewUpdate {
        requested: asset_ids.len() as u64,
        updated,
        review_status: review_status.to_owned(),
    })
}

fn read_candidates(project_path: &Path) -> Result<Vec<Candidate>, CoreError> {
    let connection = open_database(project_path)?;
    initialize_schema(&connection)?;
    let mut statement = connection.prepare(
        "SELECT ia.id, ia.role, ia.internal_path, ia.checksum_sha256,
                ia.width, ia.height,
                COALESCE(gr.source_target_path, ia.id)
         FROM image_assets ia
         LEFT JOIN generation_records gr ON gr.image_asset_id = ia.id
         WHERE ia.status = 'succeeded'
           AND ia.review_status = 'approved'
           AND ia.role IN ('generated_positive', 'background')
         ORDER BY ia.created_at, ia.id",
    )?;
    let rows = statement.query_map([], |row| {
        let width: i64 = row.get(4)?;
        let height: i64 = row.get(5)?;
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, String>(3)?,
            width,
            height,
            row.get::<_, String>(6)?,
        ))
    })?;

    let mut candidates = Vec::new();
    for row in rows {
        let (asset_id, role, internal_path, checksum, width, height, group_key) = row?;
        let width = u32::try_from(width).map_err(|_| {
            CoreError::InvalidProject("이미지 너비가 올바르지 않습니다.".to_owned())
        })?;
        let height = u32::try_from(height).map_err(|_| {
            CoreError::InvalidProject("이미지 높이가 올바르지 않습니다.".to_owned())
        })?;
        if width == 0 || height == 0 {
            return Err(CoreError::InvalidProject(format!(
                "이미지 크기가 올바르지 않습니다: {asset_id}"
            )));
        }

        let mut annotation_statement = connection.prepare(
            "SELECT class_id, x_min, y_min, x_max, y_max, source, user_modified
             FROM annotations WHERE image_asset_id = ?1 ORDER BY id",
        )?;
        let annotations = annotation_statement
            .query_map([&asset_id], |row| {
                Ok(DatasetAnnotation {
                    class_id: row.get(0)?,
                    x_min: row.get(1)?,
                    y_min: row.get(2)?,
                    x_max: row.get(3)?,
                    y_max: row.get(4)?,
                    source: row.get(5)?,
                    user_modified: row.get::<_, i64>(6)? != 0,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        candidates.push(Candidate {
            asset_id,
            role,
            internal_path,
            checksum_sha256: checksum,
            width,
            height,
            group_key,
            annotations,
        });
    }
    Ok(candidates)
}

pub fn create_dataset_version(
    project_path: impl AsRef<Path>,
    seed: i64,
) -> Result<DatasetVersionSummary, CoreError> {
    let project_path = project_path.as_ref();
    let project = read_manifest(project_path)?;
    let task_spec = current_task_spec(project_path)?;
    let mut candidates = read_candidates(project_path)?;
    if !candidates
        .iter()
        .any(|candidate| candidate.role == "generated_positive")
    {
        return Err(CoreError::InvalidProject(
            "승인된 합성 양성 이미지가 없습니다.".to_owned(),
        ));
    }

    for candidate in &candidates {
        let file = safe_project_asset(project_path, &candidate.internal_path)?;
        let checksum = sha256_file(&file)?;
        if checksum != candidate.checksum_sha256 {
            return Err(CoreError::InvalidProject(format!(
                "자산 체크섬이 변경됐습니다: {}",
                candidate.asset_id
            )));
        }
        if candidate.role == "generated_positive" && candidate.annotations.is_empty() {
            return Err(CoreError::InvalidProject(format!(
                "양성 이미지의 Bounding Box가 누락됐습니다: {}",
                candidate.asset_id
            )));
        }
        for annotation in &candidate.annotations {
            if annotation.class_id != project.class_id
                || annotation.x_min < 0
                || annotation.y_min < 0
                || annotation.x_max <= annotation.x_min
                || annotation.y_max <= annotation.y_min
                || annotation.x_max > i64::from(candidate.width)
                || annotation.y_max > i64::from(candidate.height)
            {
                return Err(CoreError::InvalidProject(format!(
                    "Bounding Box가 이미지 범위를 벗어났습니다: {}",
                    candidate.asset_id
                )));
            }
        }
    }

    let positive_splits = split_groups(
        candidates
            .iter()
            .filter(|candidate| candidate.role == "generated_positive")
            .map(|candidate| candidate.group_key.clone()),
        seed,
    );
    let negative_splits = split_groups(
        candidates
            .iter()
            .filter(|candidate| candidate.role == "background")
            .map(|candidate| candidate.group_key.clone()),
        seed ^ 0x5f37_59df,
    );

    let mut items = Vec::with_capacity(candidates.len());
    for candidate in candidates.drain(..) {
        let splits = if candidate.role == "generated_positive" {
            &positive_splits
        } else {
            &negative_splits
        };
        let split = splits
            .get(&candidate.group_key)
            .cloned()
            .ok_or_else(|| CoreError::InvalidProject("데이터 분할 계산 실패".to_owned()))?;
        items.push(DatasetItem {
            asset_id: candidate.asset_id,
            role: candidate.role,
            path: candidate.internal_path.replace('\\', "/"),
            checksum_sha256: candidate.checksum_sha256,
            width: candidate.width,
            height: candidate.height,
            split,
            group_key: candidate.group_key,
            annotations: candidate.annotations,
        });
    }
    items.sort_by(|left, right| left.asset_id.cmp(&right.asset_id));

    let stats = DatasetStats {
        total_items: items.len() as u64,
        positive_items: items
            .iter()
            .filter(|item| item.role == "generated_positive")
            .count() as u64,
        negative_items: items
            .iter()
            .filter(|item| item.role == "background")
            .count() as u64,
        train_items: items.iter().filter(|item| item.split == "train").count() as u64,
        validation_items: items
            .iter()
            .filter(|item| item.split == "validation")
            .count() as u64,
        test_items: items.iter().filter(|item| item.split == "test").count() as u64,
    };
    let mut warnings = Vec::new();
    if stats.negative_items == 0 {
        warnings.push("승인된 부정 배경이 없어 오탐 평가가 제한됩니다.".to_owned());
    }
    if stats.validation_items == 0 {
        warnings
            .push("독립 검증 그룹이 없어 학습 점수를 일반화 성능으로 볼 수 없습니다.".to_owned());
    }
    if stats.test_items == 0 {
        warnings.push("독립 테스트 그룹이 없습니다.".to_owned());
    }

    let mut connection = open_database(project_path)?;
    initialize_schema(&connection)?;
    let previous_version: Option<i64> = connection.query_row(
        "SELECT MAX(version_number) FROM dataset_versions",
        [],
        |row| row.get(0),
    )?;
    let version = previous_version.unwrap_or(0) + 1;
    let id = Uuid::new_v4().to_string();
    let created_at = Utc::now().to_rfc3339();
    let manifest = DatasetManifest {
        schema_version: 1,
        id: id.clone(),
        version: version as u64,
        project_id: project.id,
        class_id: project.class_id,
        class_name: project.class_name,
        task_spec_id: task_spec.id,
        task_spec_revision: task_spec.revision,
        generation_policy: task_spec.generation_policy,
        created_at: created_at.clone(),
        seed,
        immutable: true,
        stats: stats.clone(),
        warnings: warnings.clone(),
        items,
    };

    let relative_manifest = format!("datasets/v{version:04}/dataset.json");
    let destination = project_path.join(&relative_manifest);
    let temporary = destination.with_extension("json.tmp");
    fs::create_dir_all(destination.parent().ok_or_else(|| {
        CoreError::InvalidProject("데이터셋 경로를 만들 수 없습니다.".to_owned())
    })?)?;
    fs::write(&temporary, serde_json::to_vec_pretty(&manifest)?)?;
    fs::rename(&temporary, &destination)?;
    let checksum = sha256_file(&destination)?;

    let total_items = i64::try_from(stats.total_items)
        .map_err(|_| CoreError::InvalidProject("데이터셋 항목 수가 너무 큽니다.".to_owned()))?;
    let train_items = i64::try_from(stats.train_items)
        .map_err(|_| CoreError::InvalidProject("학습 항목 수가 너무 큽니다.".to_owned()))?;
    let validation_items = i64::try_from(stats.validation_items)
        .map_err(|_| CoreError::InvalidProject("검증 항목 수가 너무 큽니다.".to_owned()))?;
    let test_items = i64::try_from(stats.test_items)
        .map_err(|_| CoreError::InvalidProject("테스트 항목 수가 너무 큽니다.".to_owned()))?;
    let transaction = connection.transaction()?;
    transaction.execute(
        "INSERT INTO dataset_versions(
            id, version_number, status, manifest_path, checksum_sha256, seed,
            total_items, train_items, validation_items, test_items, created_at
         ) VALUES (?1, ?2, 'ready', ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
        params![
            id,
            version,
            relative_manifest,
            checksum,
            seed,
            total_items,
            train_items,
            validation_items,
            test_items,
            created_at,
        ],
    )?;
    for item in &manifest.items {
        transaction.execute(
            "INSERT INTO dataset_items(
                dataset_id, image_asset_id, split, group_key, annotations_json
             ) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                id,
                item.asset_id,
                item.split,
                item.group_key,
                serde_json::to_string(&item.annotations)?,
            ],
        )?;
    }
    transaction.commit()?;

    Ok(DatasetVersionSummary {
        id,
        version: version as u64,
        manifest_path: destination.to_string_lossy().into_owned(),
        checksum_sha256: checksum,
        created_at,
        task_spec_id: manifest.task_spec_id,
        task_spec_revision: manifest.task_spec_revision,
        stats,
        warnings,
    })
}

pub fn dataset_manifest_path(
    project_path: impl AsRef<Path>,
    dataset_id: &str,
) -> Result<PathBuf, CoreError> {
    let project_path = project_path.as_ref();
    let _manifest = read_manifest(project_path)?;
    let connection = open_database(project_path)?;
    initialize_schema(&connection)?;
    let relative: String = connection.query_row(
        "SELECT manifest_path FROM dataset_versions WHERE id = ?1 AND status = 'ready'",
        [dataset_id],
        |row| row.get(0),
    )?;
    safe_project_asset(project_path, &relative)
}

pub fn latest_dataset_version(
    project_path: impl AsRef<Path>,
) -> Result<Option<DatasetVersionSummary>, CoreError> {
    let project_path = project_path.as_ref();
    let _project = read_manifest(project_path)?;
    let connection = open_database(project_path)?;
    initialize_schema(&connection)?;
    let row: Option<(String, i64, String, String, String)> = connection
        .query_row(
            "SELECT id, version_number, manifest_path, checksum_sha256, created_at
             FROM dataset_versions WHERE status = 'ready'
             ORDER BY version_number DESC LIMIT 1",
            [],
            |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                ))
            },
        )
        .optional()?;
    let Some((id, version, relative, checksum_sha256, created_at)) = row else {
        return Ok(None);
    };
    let path = safe_project_asset(project_path, &relative)?;
    let manifest: DatasetManifest = serde_json::from_slice(&fs::read(&path)?)?;
    Ok(Some(DatasetVersionSummary {
        id,
        version: version.max(0) as u64,
        manifest_path: path.to_string_lossy().into_owned(),
        checksum_sha256,
        created_at,
        task_spec_id: manifest.task_spec_id,
        task_spec_revision: manifest.task_spec_revision,
        stats: manifest.stats,
        warnings: manifest.warnings,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        BoundingBoxInput, GeneratedImageInput, ImageInspectionInput, create_project,
        import_background_images, record_generation_results, start_job,
    };

    fn inspection(path: &Path, checksum: &str) -> ImageInspectionInput {
        ImageInspectionInput {
            path: path.to_string_lossy().into_owned(),
            status: "succeeded".to_owned(),
            checksum_sha256: Some(checksum.to_owned()),
            perceptual_hash: None,
            image_format: Some("PNG".to_owned()),
            width: Some(64),
            height: Some(64),
            file_size: Some(4),
            brightness_mean: None,
            contrast_stddev: None,
            blur_score: None,
            has_alpha: Some(false),
            warnings: Vec::new(),
            error_code: None,
            error_message: None,
        }
    }

    #[test]
    fn approved_assets_create_an_immutable_dataset() {
        let root = tempfile::tempdir().expect("temporary directory");
        let project = create_project(root.path(), "데이터셋", "부품").expect("project");

        let generated_path = Path::new(&project.path)
            .join("assets/generated")
            .join("sample.png");
        fs::write(&generated_path, b"generated").expect("generated file");
        let generated_checksum = sha256_file(&generated_path).expect("checksum");
        let job = start_job(&project.path, "synthetic_generation", 1).expect("job");
        let generated = GeneratedImageInput {
            status: "succeeded".to_owned(),
            output_path: generated_path.to_string_lossy().into_owned(),
            source_target: "target-group-a".to_owned(),
            source_background: "background-a".to_owned(),
            seed: 7,
            bounding_box: Some(BoundingBoxInput {
                x_min: 8,
                y_min: 8,
                x_max: 48,
                y_max: 48,
            }),
            checksum_sha256: Some(generated_checksum),
            width: Some(64),
            height: Some(64),
            recipe: serde_json::json!({}),
            warnings: Vec::new(),
            error_code: None,
            error_message: None,
        };
        let generated_record = record_generation_results(&project.path, &job.id, &[generated])
            .expect("record generation");

        let background_source = root.path().join("background.png");
        fs::write(&background_source, b"background").expect("background file");
        let background_checksum = sha256_file(&background_source).expect("checksum");
        let background = import_background_images(
            &project.path,
            &[inspection(&background_source, &background_checksum)],
        )
        .expect("background import");

        let ids = vec![
            generated_record.items[0].id.clone().expect("generated id"),
            background[0].id.clone(),
        ];
        let review = set_image_review_status(&project.path, &ids, "approved").expect("approve");
        assert_eq!(review.updated, 2);

        let dataset = create_dataset_version(&project.path, 2026).expect("dataset");
        assert_eq!(dataset.version, 1);
        assert_eq!(dataset.stats.positive_items, 1);
        assert_eq!(dataset.stats.negative_items, 1);
        assert!(Path::new(&dataset.manifest_path).exists());
        let manifest: DatasetManifest = serde_json::from_slice(
            &fs::read(&dataset.manifest_path).expect("read dataset manifest"),
        )
        .expect("parse manifest");
        assert!(manifest.immutable);
        assert_eq!(manifest.items.len(), 2);
    }
}
