use std::fs;
use std::path::{Path, PathBuf};

use chrono::Utc;
use rusqlite::{Connection, OpenFlags, OptionalExtension, params};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

const SCHEMA_VERSION: u32 = 6;

#[derive(Debug, Error)]
pub enum CoreError {
    #[error("프로젝트 이름을 입력해 주세요.")]
    EmptyProjectName,
    #[error("대상 클래스 이름을 입력해 주세요.")]
    EmptyClassName,
    #[error("프로젝트 파일이 올바르지 않습니다: {0}")]
    InvalidProject(String),
    #[error("파일 작업에 실패했습니다: {0}")]
    Io(#[from] std::io::Error),
    #[error("프로젝트 데이터베이스 작업에 실패했습니다: {0}")]
    Database(#[from] rusqlite::Error),
    #[error("프로젝트 메타데이터 처리에 실패했습니다: {0}")]
    Json(#[from] serde_json::Error),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProjectManifest {
    pub schema_version: u32,
    pub id: String,
    pub name: String,
    pub class_id: String,
    pub class_name: String,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProjectSummary {
    pub id: String,
    pub name: String,
    pub class_id: String,
    pub class_name: String,
    pub path: String,
    pub created_at: String,
    pub image_count: u64,
    pub warning_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WarningInput {
    pub code: String,
    pub message: String,
    pub value: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageInspectionInput {
    pub path: String,
    pub status: String,
    pub checksum_sha256: Option<String>,
    pub perceptual_hash: Option<String>,
    pub image_format: Option<String>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub file_size: Option<i64>,
    pub brightness_mean: Option<f64>,
    pub contrast_stddev: Option<f64>,
    pub blur_score: Option<f64>,
    pub has_alpha: Option<bool>,
    #[serde(default)]
    pub warnings: Vec<WarningInput>,
    pub error_code: Option<String>,
    pub error_message: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImportedImage {
    pub id: String,
    pub role: String,
    pub original_path: String,
    pub original_name: String,
    pub internal_path: Option<String>,
    pub status: String,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub checksum_sha256: Option<String>,
    pub warnings: Vec<WarningInput>,
    pub duplicate: bool,
    pub review_status: String,
    pub error_code: Option<String>,
    pub error_message: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all(serialize = "camelCase", deserialize = "snake_case"))]
pub struct BoundingBoxInput {
    pub x_min: i64,
    pub y_min: i64,
    pub x_max: i64,
    pub y_max: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedImageInput {
    pub status: String,
    pub output_path: String,
    pub source_target: String,
    pub source_background: String,
    pub seed: i64,
    pub bounding_box: Option<BoundingBoxInput>,
    pub checksum_sha256: Option<String>,
    pub width: Option<i64>,
    pub height: Option<i64>,
    #[serde(default)]
    pub recipe: serde_json::Value,
    #[serde(default)]
    pub warnings: Vec<WarningInput>,
    pub error_code: Option<String>,
    pub error_message: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeneratedImageRecord {
    pub id: Option<String>,
    pub status: String,
    pub output_path: String,
    pub seed: i64,
    pub bounding_box: Option<BoundingBoxInput>,
    pub warnings: Vec<WarningInput>,
    pub review_status: String,
    pub error_code: Option<String>,
    pub error_message: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JobSummary {
    pub id: String,
    pub job_type: String,
    pub status: String,
    pub total_items: u64,
    pub completed_items: u64,
    pub failed_items: u64,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerationBatchResult {
    pub job: JobSummary,
    pub items: Vec<GeneratedImageRecord>,
}

fn project_database_path(project_path: &Path) -> PathBuf {
    project_path.join("project.sqlite")
}

pub(crate) fn open_database(project_path: &Path) -> Result<Connection, CoreError> {
    let database_path = project_database_path(project_path);
    if !database_path.is_file() {
        return Err(CoreError::InvalidProject(
            "프로젝트 데이터베이스(project.sqlite)가 없습니다. 올바른 VisionForge 프로젝트 폴더를 선택해 주세요."
                .to_owned(),
        ));
    }
    let connection = Connection::open_with_flags(
        database_path,
        OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )?;
    configure_database(&connection)?;
    Ok(connection)
}

fn create_database(project_path: &Path) -> Result<Connection, CoreError> {
    let connection = Connection::open(project_database_path(project_path))?;
    configure_database(&connection)?;
    Ok(connection)
}

fn configure_database(connection: &Connection) -> Result<(), CoreError> {
    connection.execute_batch(
        "PRAGMA foreign_keys = ON;
         PRAGMA journal_mode = WAL;
         PRAGMA synchronous = NORMAL;
         PRAGMA busy_timeout = 5000;",
    )?;
    Ok(())
}

pub(crate) fn initialize_schema(connection: &Connection) -> Result<(), CoreError> {
    connection.execute_batch(
        "CREATE TABLE IF NOT EXISTS schema_info (
            version INTEGER NOT NULL
         );

         CREATE TABLE IF NOT EXISTS project (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            class_id TEXT NOT NULL,
            class_name TEXT NOT NULL,
            created_at TEXT NOT NULL
         );

         CREATE TABLE IF NOT EXISTS image_assets (
            id TEXT PRIMARY KEY,
            role TEXT NOT NULL,
            original_path TEXT NOT NULL,
            original_name TEXT NOT NULL,
            internal_path TEXT,
            status TEXT NOT NULL,
            checksum_sha256 TEXT,
            perceptual_hash TEXT,
            image_format TEXT,
            width INTEGER,
            height INTEGER,
            file_size INTEGER,
            brightness_mean REAL,
            contrast_stddev REAL,
            blur_score REAL,
            has_alpha INTEGER,
            warnings_json TEXT NOT NULL,
            error_code TEXT,
            error_message TEXT,
            review_status TEXT NOT NULL,
            created_at TEXT NOT NULL
         );

         CREATE INDEX IF NOT EXISTS idx_image_assets_checksum
            ON image_assets(checksum_sha256);
         CREATE INDEX IF NOT EXISTS idx_image_assets_role
            ON image_assets(role);

         CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            job_type TEXT NOT NULL,
            status TEXT NOT NULL,
            total_items INTEGER NOT NULL DEFAULT 0,
            completed_items INTEGER NOT NULL DEFAULT 0,
            failed_items INTEGER NOT NULL DEFAULT 0,
            checkpoint_json TEXT,
            error_code TEXT,
            error_message TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
         );

         CREATE TABLE IF NOT EXISTS job_items (
            id TEXT PRIMARY KEY,
            job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
            source_asset_id TEXT REFERENCES image_assets(id),
            status TEXT NOT NULL,
            error_code TEXT,
            error_message TEXT,
            result_json TEXT,
            updated_at TEXT NOT NULL
         );

         CREATE TABLE IF NOT EXISTS annotations (
            id TEXT PRIMARY KEY,
            image_asset_id TEXT NOT NULL REFERENCES image_assets(id) ON DELETE CASCADE,
            class_id TEXT NOT NULL,
            x_min INTEGER NOT NULL,
            y_min INTEGER NOT NULL,
            x_max INTEGER NOT NULL,
            y_max INTEGER NOT NULL,
            source TEXT NOT NULL,
            user_modified INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
         );

         CREATE TABLE IF NOT EXISTS generation_records (
            image_asset_id TEXT PRIMARY KEY REFERENCES image_assets(id) ON DELETE CASCADE,
            source_target_path TEXT NOT NULL,
            source_background_path TEXT NOT NULL,
            seed INTEGER NOT NULL,
            recipe_json TEXT NOT NULL
         );

         CREATE TABLE IF NOT EXISTS dataset_versions (
            id TEXT PRIMARY KEY,
            version_number INTEGER NOT NULL UNIQUE,
            status TEXT NOT NULL,
            manifest_path TEXT NOT NULL UNIQUE,
            checksum_sha256 TEXT NOT NULL,
            seed INTEGER NOT NULL,
            total_items INTEGER NOT NULL,
            train_items INTEGER NOT NULL,
            validation_items INTEGER NOT NULL,
            test_items INTEGER NOT NULL,
            created_at TEXT NOT NULL
         );

         CREATE TABLE IF NOT EXISTS dataset_items (
            dataset_id TEXT NOT NULL REFERENCES dataset_versions(id) ON DELETE CASCADE,
            image_asset_id TEXT NOT NULL REFERENCES image_assets(id),
            split TEXT NOT NULL,
            group_key TEXT NOT NULL,
            annotations_json TEXT NOT NULL,
            PRIMARY KEY(dataset_id, image_asset_id)
         );

         CREATE TABLE IF NOT EXISTS model_versions (
            id TEXT PRIMARY KEY,
            dataset_id TEXT NOT NULL REFERENCES dataset_versions(id),
            status TEXT NOT NULL,
            deployment_status TEXT NOT NULL DEFAULT 'experimental',
            engine_name TEXT NOT NULL,
            class_id TEXT NOT NULL,
            class_name TEXT NOT NULL,
            origin TEXT NOT NULL DEFAULT 'trained',
            package_path TEXT,
            model_path TEXT NOT NULL,
            model_checksum_sha256 TEXT NOT NULL,
            metrics_path TEXT NOT NULL,
            metrics_json TEXT NOT NULL,
            warnings_json TEXT NOT NULL,
            created_at TEXT NOT NULL
         );

         CREATE TABLE IF NOT EXISTS inference_runs (
            id TEXT PRIMARY KEY,
            job_id TEXT NOT NULL UNIQUE REFERENCES jobs(id),
            model_id TEXT NOT NULL REFERENCES model_versions(id),
            confidence_threshold REAL NOT NULL,
            task_spec_id TEXT,
            task_spec_revision INTEGER,
            output_policy_json TEXT,
            created_at TEXT NOT NULL
         );

         CREATE TABLE IF NOT EXISTS inference_results (
            id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL REFERENCES inference_runs(id) ON DELETE CASCADE,
            input_asset_id TEXT NOT NULL REFERENCES image_assets(id),
            output_asset_id TEXT REFERENCES image_assets(id),
            status TEXT NOT NULL,
            max_confidence REAL,
            decision TEXT,
            routed_path TEXT,
            routing_error TEXT,
            elapsed_ms REAL,
            error_code TEXT,
            error_message TEXT,
            created_at TEXT NOT NULL
         );

         CREATE TABLE IF NOT EXISTS detections (
            id TEXT PRIMARY KEY,
            inference_result_id TEXT NOT NULL REFERENCES inference_results(id) ON DELETE CASCADE,
            class_id TEXT NOT NULL,
            class_name TEXT NOT NULL,
            confidence REAL NOT NULL,
            x_min INTEGER NOT NULL,
            y_min INTEGER NOT NULL,
            x_max INTEGER NOT NULL,
            y_max INTEGER NOT NULL
         );

         CREATE TABLE IF NOT EXISTS task_specs (
            id TEXT PRIMARY KEY,
            revision INTEGER NOT NULL UNIQUE,
            task_type TEXT NOT NULL,
            pipeline_id TEXT NOT NULL,
            spec_json TEXT NOT NULL,
            created_at TEXT NOT NULL
         );",
    )?;

    let model_columns = {
        let mut statement = connection.prepare("PRAGMA table_info(model_versions)")?;
        statement
            .query_map([], |row| row.get::<_, String>(1))?
            .collect::<Result<Vec<_>, _>>()?
    };
    for (column, migration) in [
        (
            "class_id",
            "ALTER TABLE model_versions ADD COLUMN class_id TEXT",
        ),
        (
            "class_name",
            "ALTER TABLE model_versions ADD COLUMN class_name TEXT",
        ),
        (
            "origin",
            "ALTER TABLE model_versions ADD COLUMN origin TEXT NOT NULL DEFAULT 'trained'",
        ),
        (
            "package_path",
            "ALTER TABLE model_versions ADD COLUMN package_path TEXT",
        ),
        (
            "deployment_status",
            "ALTER TABLE model_versions ADD COLUMN deployment_status TEXT NOT NULL DEFAULT 'experimental'",
        ),
    ] {
        if !model_columns.iter().any(|existing| existing == column) {
            connection.execute_batch(migration)?;
        }
    }
    connection.execute_batch(
        "UPDATE model_versions
         SET class_id = COALESCE(class_id, (SELECT class_id FROM project LIMIT 1)),
             class_name = COALESCE(class_name, (SELECT class_name FROM project LIMIT 1)),
             origin = COALESCE(origin, 'trained'),
             deployment_status = COALESCE(deployment_status, 'experimental');",
    )?;

    let inference_run_columns = {
        let mut statement = connection.prepare("PRAGMA table_info(inference_runs)")?;
        statement
            .query_map([], |row| row.get::<_, String>(1))?
            .collect::<Result<Vec<_>, _>>()?
    };
    for (column, migration) in [
        (
            "task_spec_id",
            "ALTER TABLE inference_runs ADD COLUMN task_spec_id TEXT",
        ),
        (
            "task_spec_revision",
            "ALTER TABLE inference_runs ADD COLUMN task_spec_revision INTEGER",
        ),
        (
            "output_policy_json",
            "ALTER TABLE inference_runs ADD COLUMN output_policy_json TEXT",
        ),
    ] {
        if !inference_run_columns
            .iter()
            .any(|existing| existing == column)
        {
            connection.execute_batch(migration)?;
        }
    }

    let inference_result_columns = {
        let mut statement = connection.prepare("PRAGMA table_info(inference_results)")?;
        statement
            .query_map([], |row| row.get::<_, String>(1))?
            .collect::<Result<Vec<_>, _>>()?
    };
    for (column, migration) in [
        (
            "max_confidence",
            "ALTER TABLE inference_results ADD COLUMN max_confidence REAL",
        ),
        (
            "decision",
            "ALTER TABLE inference_results ADD COLUMN decision TEXT",
        ),
        (
            "routed_path",
            "ALTER TABLE inference_results ADD COLUMN routed_path TEXT",
        ),
        (
            "routing_error",
            "ALTER TABLE inference_results ADD COLUMN routing_error TEXT",
        ),
    ] {
        if !inference_result_columns
            .iter()
            .any(|existing| existing == column)
        {
            connection.execute_batch(migration)?;
        }
    }

    let version: Option<u32> = connection
        .query_row("SELECT version FROM schema_info LIMIT 1", [], |row| {
            row.get(0)
        })
        .optional()?;
    if version.is_none() {
        connection.execute(
            "INSERT INTO schema_info(version) VALUES (?1)",
            [SCHEMA_VERSION],
        )?;
    } else if version.is_some_and(|value| value < SCHEMA_VERSION) {
        connection.execute("UPDATE schema_info SET version = ?1", [SCHEMA_VERSION])?;
    }
    Ok(())
}

fn write_manifest(project_path: &Path, manifest: &ProjectManifest) -> Result<(), CoreError> {
    let destination = project_path.join("project.json");
    let temporary = project_path.join("project.json.tmp");
    fs::write(&temporary, serde_json::to_vec_pretty(manifest)?)?;
    fs::rename(temporary, destination)?;
    Ok(())
}

pub(crate) fn read_manifest(project_path: &Path) -> Result<ProjectManifest, CoreError> {
    let manifest_path = project_path.join("project.json");
    let content = match fs::read(&manifest_path) {
        Ok(content) => content,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Err(CoreError::InvalidProject(
                "선택한 폴더는 VisionForge 프로젝트가 아닙니다. project.json과 project.sqlite가 있는 기존 프로젝트 폴더를 선택하거나 첫 화면에서 새 프로젝트를 만들어 주세요."
                    .to_owned(),
            ));
        }
        Err(error) => return Err(error.into()),
    };
    let manifest: ProjectManifest = serde_json::from_slice(&content)?;
    if manifest.schema_version > SCHEMA_VERSION {
        return Err(CoreError::InvalidProject(format!(
            "지원하지 않는 스키마 버전 {}",
            manifest.schema_version
        )));
    }
    Ok(manifest)
}

fn summary_from_project(project_path: &Path) -> Result<ProjectSummary, CoreError> {
    let manifest = read_manifest(project_path)?;
    let connection = open_database(project_path)?;
    initialize_schema(&connection)?;
    let image_count: i64 = connection.query_row(
        "SELECT COUNT(*) FROM image_assets WHERE status = 'succeeded'",
        [],
        |row| row.get(0),
    )?;
    let warning_count: i64 = connection.query_row(
        "SELECT COUNT(*) FROM image_assets
         WHERE warnings_json <> '[]' OR status = 'failed'",
        [],
        |row| row.get(0),
    )?;
    Ok(ProjectSummary {
        id: manifest.id,
        name: manifest.name,
        class_id: manifest.class_id,
        class_name: manifest.class_name,
        path: project_path.to_string_lossy().into_owned(),
        created_at: manifest.created_at,
        image_count: image_count.max(0) as u64,
        warning_count: warning_count.max(0) as u64,
    })
}

pub fn create_project(
    base_directory: impl AsRef<Path>,
    name: &str,
    class_name: &str,
) -> Result<ProjectSummary, CoreError> {
    let name = name.trim();
    let class_name = class_name.trim();
    if name.is_empty() {
        return Err(CoreError::EmptyProjectName);
    }
    if class_name.is_empty() {
        return Err(CoreError::EmptyClassName);
    }

    let id = Uuid::new_v4().to_string();
    let class_id = Uuid::new_v4().to_string();
    let project_path = base_directory.as_ref().join(format!("vf-{}", &id[..8]));
    for relative in [
        "assets/originals",
        "assets/foregrounds",
        "assets/backgrounds",
        "assets/generated",
        "assets/inference-inputs",
        "assets/results",
        "thumbnails",
        "datasets",
        "models",
        "jobs",
        "exports",
        "backups",
        "logs",
        "temp",
    ] {
        fs::create_dir_all(project_path.join(relative))?;
    }

    let created_at = Utc::now().to_rfc3339();
    let manifest = ProjectManifest {
        schema_version: SCHEMA_VERSION,
        id: id.clone(),
        name: name.to_owned(),
        class_id: class_id.clone(),
        class_name: class_name.to_owned(),
        created_at: created_at.clone(),
    };
    write_manifest(&project_path, &manifest)?;

    let connection = create_database(&project_path)?;
    initialize_schema(&connection)?;
    connection.execute(
        "INSERT INTO project(id, name, class_id, class_name, created_at)
         VALUES (?1, ?2, ?3, ?4, ?5)",
        params![id, name, class_id, class_name, created_at],
    )?;
    summary_from_project(&project_path)
}

pub fn open_project(project_path: impl AsRef<Path>) -> Result<ProjectSummary, CoreError> {
    summary_from_project(project_path.as_ref())
}

fn safe_extension(path: &Path, image_format: Option<&str>) -> &'static str {
    match image_format.map(str::to_ascii_lowercase).as_deref() {
        Some("jpeg" | "jpg") => "jpg",
        Some("png") => "png",
        Some("webp") => "webp",
        Some("bmp") => "bmp",
        _ => match path.extension().and_then(|value| value.to_str()) {
            Some(value) if value.eq_ignore_ascii_case("jpeg") => "jpg",
            Some(value) if value.eq_ignore_ascii_case("png") => "png",
            Some(value) if value.eq_ignore_ascii_case("webp") => "webp",
            Some(value) if value.eq_ignore_ascii_case("bmp") => "bmp",
            _ => "img",
        },
    }
}

pub fn import_target_images(
    project_path: impl AsRef<Path>,
    inspections: &[ImageInspectionInput],
) -> Result<Vec<ImportedImage>, CoreError> {
    import_images(
        project_path.as_ref(),
        inspections,
        "target_original",
        "originals",
    )
}

pub fn import_background_images(
    project_path: impl AsRef<Path>,
    inspections: &[ImageInspectionInput],
) -> Result<Vec<ImportedImage>, CoreError> {
    import_images(
        project_path.as_ref(),
        inspections,
        "background",
        "backgrounds",
    )
}

pub fn import_inference_images(
    project_path: impl AsRef<Path>,
    inspections: &[ImageInspectionInput],
) -> Result<Vec<ImportedImage>, CoreError> {
    import_images(
        project_path.as_ref(),
        inspections,
        "inference_input",
        "inference-inputs",
    )
}

fn import_images(
    project_path: &Path,
    inspections: &[ImageInspectionInput],
    role: &str,
    storage_directory: &str,
) -> Result<Vec<ImportedImage>, CoreError> {
    let _manifest = read_manifest(project_path)?;
    let mut connection = open_database(project_path)?;
    initialize_schema(&connection)?;
    let transaction = connection.transaction()?;
    let created_at = Utc::now().to_rfc3339();
    let mut imported = Vec::with_capacity(inspections.len());

    for inspection in inspections {
        let source = PathBuf::from(&inspection.path);
        let original_name = source
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("unknown")
            .to_owned();

        if let ("succeeded", Some(checksum)) = (
            inspection.status.as_str(),
            inspection.checksum_sha256.as_deref(),
        ) {
            let existing: Option<(String, Option<String>, String)> = transaction
                .query_row(
                    "SELECT id, internal_path, review_status FROM image_assets
                         WHERE checksum_sha256 = ?1 AND role = ?2
                         LIMIT 1",
                    params![checksum, role],
                    |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
                )
                .optional()?;
            if let Some((id, internal_path, review_status)) = existing {
                let mut warnings = inspection.warnings.clone();
                warnings.push(WarningInput {
                    code: "exact_duplicate".to_owned(),
                    message: "동일한 이미지가 이미 프로젝트에 등록되어 있습니다.".to_owned(),
                    value: None,
                });
                imported.push(ImportedImage {
                    id,
                    role: role.to_owned(),
                    original_path: inspection.path.clone(),
                    original_name,
                    internal_path,
                    status: "succeeded".to_owned(),
                    width: inspection.width,
                    height: inspection.height,
                    checksum_sha256: inspection.checksum_sha256.clone(),
                    warnings,
                    duplicate: true,
                    review_status,
                    error_code: None,
                    error_message: None,
                });
                continue;
            }
        }

        let id = Uuid::new_v4().to_string();
        let internal_path = if inspection.status == "succeeded" {
            let checksum = inspection
                .checksum_sha256
                .as_deref()
                .ok_or_else(|| CoreError::InvalidProject("이미지 체크섬 누락".to_owned()))?;
            let extension = safe_extension(&source, inspection.image_format.as_deref());
            let relative = PathBuf::from("assets")
                .join(storage_directory)
                .join(format!("{checksum}.{extension}"));
            let destination = project_path.join(&relative);
            if !destination.exists() {
                fs::copy(&source, &destination)?;
            }
            Some(relative.to_string_lossy().replace('\\', "/"))
        } else {
            None
        };
        let review_status = if inspection.status == "failed" || !inspection.warnings.is_empty() {
            "needs_review"
        } else {
            "unreviewed"
        };
        transaction.execute(
            "INSERT INTO image_assets(
                id, role, original_path, original_name, internal_path, status,
                checksum_sha256, perceptual_hash, image_format, width, height,
                file_size, brightness_mean, contrast_stddev, blur_score, has_alpha,
                warnings_json, error_code, error_message, review_status, created_at
             ) VALUES (
                ?1, ?2, ?3, ?4, ?5, ?6,
                ?7, ?8, ?9, ?10, ?11,
                ?12, ?13, ?14, ?15, ?16,
                ?17, ?18, ?19, ?20, ?21
             )",
            params![
                id,
                role,
                inspection.path,
                original_name,
                internal_path,
                inspection.status,
                inspection.checksum_sha256,
                inspection.perceptual_hash,
                inspection.image_format,
                inspection.width,
                inspection.height,
                inspection.file_size,
                inspection.brightness_mean,
                inspection.contrast_stddev,
                inspection.blur_score,
                inspection.has_alpha,
                serde_json::to_string(&inspection.warnings)?,
                inspection.error_code,
                inspection.error_message,
                review_status,
                created_at,
            ],
        )?;
        imported.push(ImportedImage {
            id,
            role: role.to_owned(),
            original_path: inspection.path.clone(),
            original_name,
            internal_path,
            status: inspection.status.clone(),
            width: inspection.width,
            height: inspection.height,
            checksum_sha256: inspection.checksum_sha256.clone(),
            warnings: inspection.warnings.clone(),
            duplicate: false,
            review_status: review_status.to_owned(),
            error_code: inspection.error_code.clone(),
            error_message: inspection.error_message.clone(),
        });
    }
    transaction.commit()?;
    Ok(imported)
}

pub fn list_image_paths(
    project_path: impl AsRef<Path>,
    role: &str,
) -> Result<Vec<String>, CoreError> {
    let project_path = project_path.as_ref();
    let _manifest = read_manifest(project_path)?;
    let connection = open_database(project_path)?;
    let mut statement = connection.prepare(
        "SELECT internal_path FROM image_assets
         WHERE role = ?1 AND status = 'succeeded' AND internal_path IS NOT NULL
         ORDER BY created_at, id",
    )?;
    let paths = statement
        .query_map([role], |row| row.get::<_, String>(0))?
        .map(|row| row.map(|relative| project_path.join(relative).to_string_lossy().into_owned()))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(paths)
}

pub fn list_imported_images(
    project_path: impl AsRef<Path>,
    role: &str,
) -> Result<Vec<ImportedImage>, CoreError> {
    let project_path = project_path.as_ref();
    let _manifest = read_manifest(project_path)?;
    let connection = open_database(project_path)?;
    initialize_schema(&connection)?;
    let mut statement = connection.prepare(
        "SELECT id, role, original_path, original_name, internal_path, status,
                width, height, checksum_sha256, warnings_json, review_status,
                error_code, error_message
         FROM image_assets WHERE role = ?1 ORDER BY created_at, id",
    )?;
    let values = statement
        .query_map([role], |row| {
            let width: Option<i64> = row.get(6)?;
            let height: Option<i64> = row.get(7)?;
            let warnings_json: String = row.get(9)?;
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, Option<String>>(4)?,
                row.get::<_, String>(5)?,
                width,
                height,
                row.get::<_, Option<String>>(8)?,
                warnings_json,
                row.get::<_, String>(10)?,
                row.get::<_, Option<String>>(11)?,
                row.get::<_, Option<String>>(12)?,
            ))
        })?
        .collect::<Result<Vec<_>, _>>()?;
    values
        .into_iter()
        .map(
            |(
                id,
                role,
                original_path,
                original_name,
                internal_path,
                status,
                width,
                height,
                checksum_sha256,
                warnings_json,
                review_status,
                error_code,
                error_message,
            )| {
                Ok(ImportedImage {
                    id,
                    role,
                    original_path,
                    original_name,
                    internal_path,
                    status,
                    width: width.and_then(|value| u32::try_from(value).ok()),
                    height: height.and_then(|value| u32::try_from(value).ok()),
                    checksum_sha256,
                    warnings: serde_json::from_str(&warnings_json)?,
                    duplicate: false,
                    review_status,
                    error_code,
                    error_message,
                })
            },
        )
        .collect()
}

pub fn list_generated_images(
    project_path: impl AsRef<Path>,
) -> Result<Vec<GeneratedImageRecord>, CoreError> {
    let project_path = project_path.as_ref();
    let _manifest = read_manifest(project_path)?;
    let connection = open_database(project_path)?;
    initialize_schema(&connection)?;
    let mut statement = connection.prepare(
        "SELECT ia.id, ia.internal_path, gr.seed,
                a.x_min, a.y_min, a.x_max, a.y_max,
                ia.warnings_json, ia.review_status, ia.error_code, ia.error_message
         FROM image_assets ia
         JOIN generation_records gr ON gr.image_asset_id = ia.id
         LEFT JOIN annotations a ON a.image_asset_id = ia.id
         WHERE ia.role = 'generated_positive' AND ia.status = 'succeeded'
         ORDER BY ia.created_at, ia.id",
    )?;
    let values = statement
        .query_map([], |row| {
            let internal_path: String = row.get(1)?;
            let x_min: Option<i64> = row.get(3)?;
            let warnings_json: String = row.get(7)?;
            Ok((
                row.get::<_, String>(0)?,
                internal_path,
                row.get::<_, i64>(2)?,
                x_min,
                row.get::<_, Option<i64>>(4)?,
                row.get::<_, Option<i64>>(5)?,
                row.get::<_, Option<i64>>(6)?,
                warnings_json,
                row.get::<_, String>(8)?,
                row.get::<_, Option<String>>(9)?,
                row.get::<_, Option<String>>(10)?,
            ))
        })?
        .collect::<Result<Vec<_>, _>>()?;
    values
        .into_iter()
        .map(
            |(
                id,
                internal_path,
                seed,
                x_min,
                y_min,
                x_max,
                y_max,
                warnings_json,
                review_status,
                error_code,
                error_message,
            )| {
                let bounding_box = match (x_min, y_min, x_max, y_max) {
                    (Some(x_min), Some(y_min), Some(x_max), Some(y_max)) => {
                        Some(BoundingBoxInput {
                            x_min,
                            y_min,
                            x_max,
                            y_max,
                        })
                    }
                    _ => None,
                };
                Ok(GeneratedImageRecord {
                    id: Some(id),
                    status: "succeeded".to_owned(),
                    output_path: project_path
                        .join(internal_path)
                        .to_string_lossy()
                        .into_owned(),
                    seed,
                    bounding_box,
                    warnings: serde_json::from_str(&warnings_json)?,
                    review_status,
                    error_code,
                    error_message,
                })
            },
        )
        .collect()
}

pub fn update_bounding_box(
    project_path: impl AsRef<Path>,
    asset_id: &str,
    bounding_box: &BoundingBoxInput,
) -> Result<BoundingBoxInput, CoreError> {
    let project_path = project_path.as_ref();
    let _manifest = read_manifest(project_path)?;
    let mut connection = open_database(project_path)?;
    initialize_schema(&connection)?;
    let dimensions: Option<(i64, i64)> = connection
        .query_row(
            "SELECT width, height FROM image_assets
             WHERE id = ?1 AND role = 'generated_positive' AND status = 'succeeded'",
            [asset_id],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .optional()?;
    let (width, height) = dimensions
        .ok_or_else(|| CoreError::InvalidProject("수정할 합성 이미지가 없습니다.".to_owned()))?;
    if bounding_box.x_min < 0
        || bounding_box.y_min < 0
        || bounding_box.x_max <= bounding_box.x_min
        || bounding_box.y_max <= bounding_box.y_min
        || bounding_box.x_max > width
        || bounding_box.y_max > height
    {
        return Err(CoreError::InvalidProject(
            "Bounding Box가 이미지 범위를 벗어났습니다.".to_owned(),
        ));
    }
    let transaction = connection.transaction()?;
    let updated = transaction.execute(
        "UPDATE annotations
         SET x_min = ?2, y_min = ?3, x_max = ?4, y_max = ?5,
             source = 'user', user_modified = 1
         WHERE image_asset_id = ?1",
        params![
            asset_id,
            bounding_box.x_min,
            bounding_box.y_min,
            bounding_box.x_max,
            bounding_box.y_max,
        ],
    )?;
    if updated == 0 {
        return Err(CoreError::InvalidProject(
            "수정할 Bounding Box가 없습니다.".to_owned(),
        ));
    }
    transaction.execute(
        "UPDATE image_assets SET review_status = 'needs_review' WHERE id = ?1",
        [asset_id],
    )?;
    transaction.commit()?;
    Ok(bounding_box.clone())
}

fn read_job(connection: &Connection, job_id: &str) -> Result<JobSummary, CoreError> {
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

pub fn start_job(
    project_path: impl AsRef<Path>,
    job_type: &str,
    total_items: u64,
) -> Result<JobSummary, CoreError> {
    let project_path = project_path.as_ref();
    let _manifest = read_manifest(project_path)?;
    let connection = open_database(project_path)?;
    initialize_schema(&connection)?;
    let id = Uuid::new_v4().to_string();
    let now = Utc::now().to_rfc3339();
    let total = i64::try_from(total_items)
        .map_err(|_| CoreError::InvalidProject("작업 항목 수가 너무 큽니다.".to_owned()))?;
    connection.execute(
        "INSERT INTO jobs(
            id, job_type, status, total_items, completed_items, failed_items,
            created_at, updated_at
         ) VALUES (?1, ?2, 'running', ?3, 0, 0, ?4, ?4)",
        params![id, job_type, total, now],
    )?;
    read_job(&connection, &id)
}

pub fn fail_job(
    project_path: impl AsRef<Path>,
    job_id: &str,
    error_code: &str,
    error_message: &str,
) -> Result<JobSummary, CoreError> {
    let project_path = project_path.as_ref();
    let connection = open_database(project_path)?;
    let now = Utc::now().to_rfc3339();
    connection.execute(
        "UPDATE jobs
         SET status = 'failed', failed_items = total_items,
             error_code = ?2, error_message = ?3, updated_at = ?4
         WHERE id = ?1",
        params![job_id, error_code, error_message, now],
    )?;
    read_job(&connection, job_id)
}

pub fn recover_interrupted_jobs(project_path: impl AsRef<Path>) -> Result<u64, CoreError> {
    let project_path = project_path.as_ref();
    let _manifest = read_manifest(project_path)?;
    let connection = open_database(project_path)?;
    initialize_schema(&connection)?;
    let now = Utc::now().to_rfc3339();
    let updated = connection.execute(
        "UPDATE jobs
         SET status = 'interrupted', error_code = 'app_interrupted',
             error_message = '앱 종료로 작업이 중단됐습니다. 완료된 파일과 기록은 보존됩니다.',
             updated_at = ?1
         WHERE status = 'running'",
        [now],
    )?;
    Ok(updated as u64)
}

pub fn record_generation_results(
    project_path: impl AsRef<Path>,
    job_id: &str,
    items: &[GeneratedImageInput],
) -> Result<GenerationBatchResult, CoreError> {
    let project_path = project_path.as_ref();
    let manifest = read_manifest(project_path)?;
    let project_canonical = fs::canonicalize(project_path)?;
    let mut connection = open_database(project_path)?;
    initialize_schema(&connection)?;
    let transaction = connection.transaction()?;
    let now = Utc::now().to_rfc3339();
    let mut completed = 0_i64;
    let mut failed = 0_i64;
    let mut records = Vec::with_capacity(items.len());

    for item in items {
        let job_item_id = Uuid::new_v4().to_string();
        let mut asset_id = None;
        if item.status == "succeeded" {
            let output_canonical = fs::canonicalize(&item.output_path)?;
            let relative = output_canonical
                .strip_prefix(&project_canonical)
                .map_err(|_| {
                    CoreError::InvalidProject(
                        "생성 결과가 프로젝트 저장소 밖에 기록됐습니다.".to_owned(),
                    )
                })?;
            let relative_path = relative.to_string_lossy().replace('\\', "/");
            let output_name = output_canonical
                .file_name()
                .and_then(|value| value.to_str())
                .unwrap_or("generated.png")
                .to_owned();
            let checksum = item.checksum_sha256.as_deref().ok_or_else(|| {
                CoreError::InvalidProject("생성 결과 체크섬이 누락됐습니다.".to_owned())
            })?;
            let bounding_box = item.bounding_box.as_ref().ok_or_else(|| {
                CoreError::InvalidProject("생성 결과 Bounding Box가 누락됐습니다.".to_owned())
            })?;
            let id = Uuid::new_v4().to_string();
            let review_status = if item.warnings.is_empty() {
                "unreviewed"
            } else {
                "needs_review"
            };
            let file_size = i64::try_from(fs::metadata(&output_canonical)?.len())
                .map_err(|_| CoreError::InvalidProject("생성 파일이 너무 큽니다.".to_owned()))?;
            transaction.execute(
                "INSERT INTO image_assets(
                    id, role, original_path, original_name, internal_path, status,
                    checksum_sha256, image_format, width, height, file_size, has_alpha,
                    warnings_json, review_status, created_at
                 ) VALUES (
                    ?1, 'generated_positive', ?2, ?3, ?4, 'succeeded',
                    ?5, 'PNG', ?6, ?7, ?8, 0,
                    ?9, ?10, ?11
                 )",
                params![
                    id,
                    item.output_path,
                    output_name,
                    relative_path,
                    checksum,
                    item.width,
                    item.height,
                    file_size,
                    serde_json::to_string(&item.warnings)?,
                    review_status,
                    now,
                ],
            )?;
            transaction.execute(
                "INSERT INTO annotations(
                    id, image_asset_id, class_id, x_min, y_min, x_max, y_max,
                    source, user_modified, created_at
                 ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, 'synthetic_mask', 0, ?8)",
                params![
                    Uuid::new_v4().to_string(),
                    id,
                    manifest.class_id,
                    bounding_box.x_min,
                    bounding_box.y_min,
                    bounding_box.x_max,
                    bounding_box.y_max,
                    now,
                ],
            )?;
            transaction.execute(
                "INSERT INTO generation_records(
                    image_asset_id, source_target_path, source_background_path,
                    seed, recipe_json
                 ) VALUES (?1, ?2, ?3, ?4, ?5)",
                params![
                    id,
                    item.source_target,
                    item.source_background,
                    item.seed,
                    serde_json::to_string(&item.recipe)?,
                ],
            )?;
            asset_id = Some(id);
            completed += 1;
        } else {
            failed += 1;
        }

        transaction.execute(
            "INSERT INTO job_items(
                id, job_id, source_asset_id, status, error_code, error_message,
                result_json, updated_at
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                job_item_id,
                job_id,
                asset_id,
                item.status,
                item.error_code,
                item.error_message,
                serde_json::to_string(item)?,
                now,
            ],
        )?;
        records.push(GeneratedImageRecord {
            id: asset_id,
            status: item.status.clone(),
            output_path: item.output_path.clone(),
            seed: item.seed,
            bounding_box: item.bounding_box.clone(),
            warnings: item.warnings.clone(),
            review_status: if item.status == "succeeded" {
                if item.warnings.is_empty() {
                    "unreviewed".to_owned()
                } else {
                    "needs_review".to_owned()
                }
            } else {
                "needs_review".to_owned()
            },
            error_code: item.error_code.clone(),
            error_message: item.error_message.clone(),
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
        "UPDATE jobs
         SET status = ?2, completed_items = ?3, failed_items = ?4, updated_at = ?5
         WHERE id = ?1",
        params![job_id, status, completed, failed, now],
    )?;
    transaction.commit()?;
    let job = read_job(&connection, job_id)?;
    Ok(GenerationBatchResult {
        job,
        items: records,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn successful_inspection(path: &Path, checksum: &str) -> ImageInspectionInput {
        ImageInspectionInput {
            path: path.to_string_lossy().into_owned(),
            status: "succeeded".to_owned(),
            checksum_sha256: Some(checksum.to_owned()),
            perceptual_hash: Some("0011223344556677".to_owned()),
            image_format: Some("PNG".to_owned()),
            width: Some(320),
            height: Some(240),
            file_size: Some(4),
            brightness_mean: Some(128.0),
            contrast_stddev: Some(40.0),
            blur_score: Some(120.0),
            has_alpha: Some(false),
            warnings: Vec::new(),
            error_code: None,
            error_message: None,
        }
    }

    #[test]
    fn project_can_be_created_and_opened() {
        let root = tempfile::tempdir().expect("temporary directory");
        let project = create_project(root.path(), "매장 제품", "빨간 상자").expect("create");

        assert_eq!(project.name, "매장 제품");
        assert_eq!(project.class_name, "빨간 상자");
        assert!(Path::new(&project.path).join("project.sqlite").exists());
        assert!(Path::new(&project.path).join("assets/originals").is_dir());

        let reopened = open_project(&project.path).expect("open");
        assert_eq!(reopened.id, project.id);
        assert_eq!(reopened.image_count, 0);

        let job = start_job(&project.path, "long_running", 3).expect("start job");
        assert_eq!(recover_interrupted_jobs(&project.path).expect("recover"), 1);
        let connection = open_database(Path::new(&project.path)).expect("database");
        assert_eq!(
            read_job(&connection, &job.id).expect("job").status,
            "interrupted"
        );
    }

    #[test]
    fn empty_directory_is_rejected_as_a_non_project() {
        let root = tempfile::tempdir().expect("temporary directory");

        let open_error = open_project(root.path()).expect_err("empty directory must be rejected");
        assert!(matches!(
            open_error,
            CoreError::InvalidProject(message)
                if message.contains("VisionForge 프로젝트가 아닙니다")
                    && message.contains("새 프로젝트")
        ));

        let recover_error = recover_interrupted_jobs(root.path())
            .expect_err("recovery must reject an empty directory");
        assert!(matches!(recover_error, CoreError::InvalidProject(_)));
        assert!(!root.path().join("project.sqlite").exists());
    }

    #[test]
    fn missing_database_is_rejected_without_recreating_it() {
        let root = tempfile::tempdir().expect("temporary directory");
        let project = create_project(root.path(), "손상 확인", "부품").expect("create");
        let database_path = Path::new(&project.path).join("project.sqlite");
        fs::remove_file(&database_path).expect("remove database");

        let error = open_project(&project.path).expect_err("missing database must be rejected");
        assert!(matches!(
            error,
            CoreError::InvalidProject(message) if message.contains("project.sqlite")
        ));
        let recover_error = recover_interrupted_jobs(&project.path)
            .expect_err("recovery must not recreate a missing database");
        assert!(matches!(recover_error, CoreError::InvalidProject(_)));
        assert!(!database_path.exists());
    }

    #[test]
    fn duplicate_import_reuses_the_original_asset() {
        let root = tempfile::tempdir().expect("temporary directory");
        let project = create_project(root.path(), "검사", "부품").expect("create");
        let source = root.path().join("source.png");
        fs::write(&source, b"test").expect("source image");
        let inspection = successful_inspection(&source, "abc123");

        let first = import_target_images(&project.path, std::slice::from_ref(&inspection))
            .expect("first import");
        let second = import_target_images(&project.path, &[inspection]).expect("second import");

        assert!(!first[0].duplicate);
        assert!(second[0].duplicate);
        assert_eq!(first[0].id, second[0].id);
        assert_eq!(open_project(&project.path).expect("summary").image_count, 1);
    }

    #[test]
    fn generation_job_records_provenance_and_annotation() {
        let root = tempfile::tempdir().expect("temporary directory");
        let project = create_project(root.path(), "합성", "부품").expect("create");
        let job = start_job(&project.path, "synthetic_generation", 1).expect("job");
        let output = Path::new(&project.path)
            .join("assets/generated")
            .join("generated-000001.png");
        fs::write(&output, b"generated-image").expect("generated output");
        let item = GeneratedImageInput {
            status: "succeeded".to_owned(),
            output_path: output.to_string_lossy().into_owned(),
            source_target: "target.png".to_owned(),
            source_background: "background.png".to_owned(),
            seed: 2026,
            bounding_box: Some(BoundingBoxInput {
                x_min: 10,
                y_min: 20,
                x_max: 80,
                y_max: 90,
            }),
            checksum_sha256: Some("generated-checksum".to_owned()),
            width: Some(160),
            height: Some(120),
            recipe: serde_json::json!({"rotation": 5}),
            warnings: Vec::new(),
            error_code: None,
            error_message: None,
        };

        let result =
            record_generation_results(&project.path, &job.id, &[item]).expect("record generation");

        assert_eq!(result.job.status, "succeeded");
        assert_eq!(result.job.completed_items, 1);
        assert!(result.items[0].id.is_some());
        assert_eq!(
            list_image_paths(&project.path, "generated_positive")
                .expect("generated paths")
                .len(),
            1
        );
        let asset_id = result.items[0].id.as_deref().expect("asset id");
        update_bounding_box(
            &project.path,
            asset_id,
            &BoundingBoxInput {
                x_min: 12,
                y_min: 22,
                x_max: 78,
                y_max: 88,
            },
        )
        .expect("update bounding box");
        let restored = list_generated_images(&project.path).expect("generated records");
        assert_eq!(
            restored[0]
                .bounding_box
                .as_ref()
                .expect("bounding box")
                .x_min,
            12
        );
        assert_eq!(restored[0].review_status, "needs_review");
    }
}
