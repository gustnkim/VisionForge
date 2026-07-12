use std::fs;
use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::model::{InferenceBatchResult, InferenceItemRecord, RoutingSummary};
use crate::project::{CoreError, initialize_schema, open_database, read_manifest};
use crate::task::{OutputPolicy, TaskSpec, validate_output_policy};

#[derive(Debug)]
struct RouteSource {
    status: String,
    max_confidence: Option<f64>,
    original_path: String,
    original_name: String,
    internal_path: Option<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RoutingManifest<'a> {
    schema_version: u32,
    run_id: &'a str,
    model_id: &'a str,
    deployment_status: &'a str,
    task_spec_id: &'a str,
    task_spec_revision: u64,
    output_policy: &'a OutputPolicy,
    items: &'a [InferenceItemRecord],
}

fn decision_for(status: &str, max_confidence: Option<f64>, policy: &OutputPolicy) -> &'static str {
    if status != "succeeded" {
        return "failed";
    }
    match max_confidence {
        Some(value) if value >= policy.positive_threshold => "present",
        Some(value) if value <= policy.negative_threshold => "absent",
        _ => "review",
    }
}

fn deployment_decision_for(
    deployment_status: &str,
    status: &str,
    max_confidence: Option<f64>,
    policy: &OutputPolicy,
) -> &'static str {
    if deployment_status != "qualified" && status == "succeeded" {
        "review"
    } else {
        decision_for(status, max_confidence, policy)
    }
}

fn folder_for<'a>(decision: &str, policy: &'a OutputPolicy) -> &'a str {
    match decision {
        "present" => &policy.present_folder,
        "absent" => &policy.absent_folder,
        "review" => &policy.review_folder,
        _ => &policy.failed_folder,
    }
}

fn safe_file_name(original_name: &str, item_id: &str) -> String {
    let fallback = format!("image-{}.bin", &item_id[..item_id.len().min(8)]);
    let raw = Path::new(original_name)
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or(&fallback);
    let sanitized = raw
        .chars()
        .map(|character| {
            if character.is_control() || r#"<>:"/\|?*"#.contains(character) {
                '_'
            } else {
                character
            }
        })
        .collect::<String>();
    let sanitized = sanitized.trim().trim_end_matches(['.', ' ']);
    if sanitized.is_empty() || sanitized == "." || sanitized == ".." {
        fallback
    } else {
        sanitized.to_owned()
    }
}

fn unique_destination(directory: &Path, file_name: &str, item_id: &str) -> PathBuf {
    let direct = directory.join(file_name);
    if !direct.exists() {
        return direct;
    }
    let path = Path::new(file_name);
    let stem = path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("image");
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .map(|value| format!(".{value}"))
        .unwrap_or_default();
    let suffix = &item_id[..item_id.len().min(8)];
    let candidate = directory.join(format!("{stem}__{suffix}{extension}"));
    if !candidate.exists() {
        return candidate;
    }
    for index in 2..=10_000_u32 {
        let candidate = directory.join(format!("{stem}__{suffix}-{index}{extension}"));
        if !candidate.exists() {
            return candidate;
        }
    }
    directory.join(format!("{stem}__{item_id}{extension}"))
}

fn checked_source(project_path: &Path, source: &RouteSource) -> Result<PathBuf, String> {
    if let Some(relative) = source.internal_path.as_deref() {
        let project = fs::canonicalize(project_path)
            .map_err(|error| format!("프로젝트 경로 확인 실패: {error}"))?;
        let candidate = fs::canonicalize(project_path.join(relative))
            .map_err(|error| format!("내부 입력 파일 확인 실패: {error}"))?;
        candidate
            .strip_prefix(&project)
            .map_err(|_| "내부 입력 파일이 프로젝트 경계를 벗어났습니다.".to_owned())?;
        return Ok(candidate);
    }
    fs::canonicalize(&source.original_path)
        .map_err(|error| format!("원본 입력 파일 확인 실패: {error}"))
}

fn copy_to_route(
    project_path: &Path,
    directory: &Path,
    item_id: &str,
    source: &RouteSource,
) -> Result<String, String> {
    let source_path = checked_source(project_path, source)?;
    fs::create_dir_all(directory).map_err(|error| format!("결과 폴더 생성 실패: {error}"))?;
    let file_name = safe_file_name(&source.original_name, item_id);
    let destination = unique_destination(directory, &file_name, item_id);
    let temporary = destination.with_extension(format!(
        "{}routing-tmp",
        destination
            .extension()
            .and_then(|value| value.to_str())
            .map(|value| format!("{value}."))
            .unwrap_or_default()
    ));
    fs::copy(&source_path, &temporary).map_err(|error| format!("결과 파일 복사 실패: {error}"))?;
    fs::rename(&temporary, &destination)
        .map_err(|error| format!("결과 파일 확정 실패: {error}"))?;
    Ok(destination.to_string_lossy().into_owned())
}

pub fn route_inference_results(
    project_path: impl AsRef<Path>,
    batch: &mut InferenceBatchResult,
    task_spec: &TaskSpec,
) -> Result<RoutingSummary, CoreError> {
    let project_path = project_path.as_ref();
    let _project = read_manifest(project_path)?;
    let policy = validate_output_policy(&task_spec.output_policy)?;
    let root = project_path.join("exports/classified").join(&batch.job.id);
    for folder in [
        &policy.present_folder,
        &policy.absent_folder,
        &policy.review_folder,
        &policy.failed_folder,
    ] {
        fs::create_dir_all(root.join(folder))?;
    }

    let mut connection = open_database(project_path)?;
    initialize_schema(&connection)?;
    let deployment_status: String = connection.query_row(
        "SELECT deployment_status FROM model_versions WHERE id = ?1 AND status = 'ready'",
        [&batch.model_id],
        |row| row.get(0),
    )?;
    let mut updates = Vec::with_capacity(batch.items.len());
    let mut present_count = 0_u64;
    let mut absent_count = 0_u64;
    let mut review_count = 0_u64;
    let mut failed_count = 0_u64;

    for item in &mut batch.items {
        let source = connection.query_row(
            "SELECT ir.status, ir.max_confidence, ia.original_path,
                    ia.original_name, ia.internal_path
             FROM inference_results ir
             JOIN image_assets ia ON ia.id = ir.input_asset_id
             WHERE ir.id = ?1 AND ir.run_id = ?2",
            rusqlite::params![item.id, batch.run_id],
            |row| {
                Ok(RouteSource {
                    status: row.get(0)?,
                    max_confidence: row.get(1)?,
                    original_path: row.get(2)?,
                    original_name: row.get(3)?,
                    internal_path: row.get(4)?,
                })
            },
        )?;
        let intended_decision = deployment_decision_for(
            &deployment_status,
            &source.status,
            source.max_confidence,
            &policy,
        );
        let directory = root.join(folder_for(intended_decision, &policy));
        let (decision, routed_path, routing_error) =
            match copy_to_route(project_path, &directory, &item.id, &source) {
                Ok(path) => (intended_decision.to_owned(), Some(path), None),
                Err(error) => ("failed".to_owned(), None, Some(error)),
            };
        match decision.as_str() {
            "present" => present_count += 1,
            "absent" => absent_count += 1,
            "review" => review_count += 1,
            _ => failed_count += 1,
        }
        item.max_confidence = source.max_confidence;
        item.decision = decision.clone();
        item.routed_path = routed_path.clone();
        item.routing_error = routing_error.clone();
        updates.push((item.id.clone(), decision, routed_path, routing_error));
    }

    let manifest_path = root.join("routing-manifest.json");
    let temporary_manifest = root.join("routing-manifest.json.tmp");
    let manifest = RoutingManifest {
        schema_version: 1,
        run_id: &batch.run_id,
        model_id: &batch.model_id,
        deployment_status: &deployment_status,
        task_spec_id: &task_spec.id,
        task_spec_revision: task_spec.revision,
        output_policy: &policy,
        items: &batch.items,
    };
    let mut serialized = serde_json::to_vec_pretty(&manifest)?;
    serialized.push(b'\n');
    fs::write(&temporary_manifest, serialized)?;
    fs::rename(&temporary_manifest, &manifest_path)?;

    let transaction = connection.transaction()?;
    for (id, decision, routed_path, routing_error) in updates {
        transaction.execute(
            "UPDATE inference_results
             SET decision = ?2, routed_path = ?3, routing_error = ?4
             WHERE id = ?1",
            rusqlite::params![id, decision, routed_path, routing_error],
        )?;
    }
    transaction.commit()?;

    let summary = RoutingSummary {
        root_path: root.to_string_lossy().into_owned(),
        manifest_path: manifest_path.to_string_lossy().into_owned(),
        present_count,
        absent_count,
        review_count,
        failed_count,
    };
    batch.routing = Some(summary.clone());
    Ok(summary)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decision_uses_a_review_band() {
        let policy = OutputPolicy::default();
        assert_eq!(decision_for("failed", None, &policy), "failed");
        assert_eq!(decision_for("succeeded", Some(0.9), &policy), "present");
        assert_eq!(decision_for("succeeded", Some(0.2), &policy), "absent");
        assert_eq!(decision_for("succeeded", Some(0.6), &policy), "review");
        assert_eq!(decision_for("succeeded", None, &policy), "review");
    }

    #[test]
    fn candidate_model_never_routes_to_an_automatic_folder() {
        let policy = OutputPolicy::default();
        assert_eq!(
            deployment_decision_for("candidate", "succeeded", Some(0.99), &policy),
            "review"
        );
        assert_eq!(
            deployment_decision_for("qualified", "succeeded", Some(0.99), &policy),
            "present"
        );
        assert_eq!(
            deployment_decision_for("candidate", "failed", None, &policy),
            "failed"
        );
    }

    #[test]
    fn duplicate_names_get_a_stable_suffix() {
        let root = tempfile::tempdir().expect("temporary directory");
        fs::write(root.path().join("photo.jpg"), b"first").expect("fixture");
        let destination = unique_destination(root.path(), "photo.jpg", "12345678-abcd");
        assert_eq!(
            destination.file_name().and_then(|value| value.to_str()),
            Some("photo__12345678.jpg")
        );
    }
}
