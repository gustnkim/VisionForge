mod engine;

use serde::Serialize;
use tauri::Manager;
use visionforge_core::{
    DatasetVersionSummary, GenerationBatchResult, ImportedImage, InferenceBatchResult,
    InferenceImageInput, ModelPackageResultInput, ModelVersionSummary, ProjectSummary,
    ReviewUpdate, TaskSpec, TaskSpecInput,
};

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SystemStatus {
    offline: bool,
    engine_ready: bool,
    engine_path: String,
    platform: &'static str,
    hardware: Option<engine::HardwareProfile>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ImportedModelProject {
    project: ProjectSummary,
    model: ModelVersionSummary,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ProjectWorkspace {
    project: ProjectSummary,
    targets: Vec<ImportedImage>,
    backgrounds: Vec<ImportedImage>,
    generated: Vec<visionforge_core::GeneratedImageRecord>,
    dataset: Option<DatasetVersionSummary>,
    model: Option<ModelVersionSummary>,
    task_spec: TaskSpec,
    recovered_jobs: u64,
}

#[tauri::command]
fn system_status() -> SystemStatus {
    SystemStatus {
        offline: true,
        engine_ready: engine::engine_ready(),
        engine_path: engine::engine_display_path(),
        platform: std::env::consts::OS,
        hardware: engine::hardware_profile(None).ok(),
    }
}

#[tauri::command]
async fn create_project(
    base_directory: String,
    name: String,
    class_name: String,
) -> Result<ProjectSummary, String> {
    tauri::async_runtime::spawn_blocking(move || {
        visionforge_core::create_project(base_directory, &name, &class_name)
            .map_err(|error| error.to_string())
    })
    .await
    .map_err(|error| format!("프로젝트 생성 작업이 중단됐습니다: {error}"))?
}

#[tauri::command]
async fn open_project_workspace(project_path: String) -> Result<ProjectWorkspace, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let recovered_jobs = visionforge_core::recover_interrupted_jobs(&project_path)
            .map_err(|error| error.to_string())?;
        let project =
            visionforge_core::open_project(&project_path).map_err(|error| error.to_string())?;
        let targets = visionforge_core::list_imported_images(&project_path, "target_original")
            .map_err(|error| error.to_string())?;
        let backgrounds = visionforge_core::list_imported_images(&project_path, "background")
            .map_err(|error| error.to_string())?;
        let generated = visionforge_core::list_generated_images(&project_path)
            .map_err(|error| error.to_string())?;
        let dataset = visionforge_core::latest_dataset_version(&project_path)
            .map_err(|error| error.to_string())?;
        let model = visionforge_core::latest_model_version(&project_path)
            .map_err(|error| error.to_string())?;
        let task_spec = visionforge_core::current_task_spec(&project_path)
            .map_err(|error| error.to_string())?;
        Ok(ProjectWorkspace {
            project,
            targets,
            backgrounds,
            generated,
            dataset,
            model,
            task_spec,
            recovered_jobs,
        })
    })
    .await
    .map_err(|error| format!("프로젝트 열기 작업이 중단됐습니다: {error}"))?
}

#[tauri::command]
async fn save_task_spec(project_path: String, input: TaskSpecInput) -> Result<TaskSpec, String> {
    tauri::async_runtime::spawn_blocking(move || {
        visionforge_core::save_task_spec(&project_path, &input).map_err(|error| error.to_string())
    })
    .await
    .map_err(|error| format!("작업 명세 저장이 중단됐습니다: {error}"))?
}

#[tauri::command]
async fn import_target_images(
    project_path: String,
    paths: Vec<String>,
) -> Result<Vec<ImportedImage>, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let inspections = engine::inspect_images(&paths)?;
        visionforge_core::import_target_images(project_path, &inspections)
            .map_err(|error| error.to_string())
    })
    .await
    .map_err(|error| format!("이미지 등록 작업이 중단됐습니다: {error}"))?
}

#[tauri::command]
async fn import_background_images(
    project_path: String,
    paths: Vec<String>,
) -> Result<Vec<ImportedImage>, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let inspections = engine::inspect_images(&paths)?;
        visionforge_core::import_background_images(project_path, &inspections)
            .map_err(|error| error.to_string())
    })
    .await
    .map_err(|error| format!("배경 이미지 등록 작업이 중단됐습니다: {error}"))?
}

#[tauri::command]
async fn generate_synthetic_batch(
    project_path: String,
    count: u64,
    seed: i64,
) -> Result<GenerationBatchResult, String> {
    tauri::async_runtime::spawn_blocking(move || {
        if count == 0 || count > 10_000 {
            return Err("한 작업의 생성 수는 1개 이상 10,000개 이하여야 합니다.".to_owned());
        }
        let targets = visionforge_core::list_image_paths(&project_path, "target_original")
            .map_err(|error| error.to_string())?;
        let backgrounds = visionforge_core::list_image_paths(&project_path, "background")
            .map_err(|error| error.to_string())?;
        let task_spec = visionforge_core::current_task_spec(&project_path)
            .map_err(|error| error.to_string())?;
        if targets.is_empty() {
            return Err("승인 가능한 대상 이미지가 없습니다.".to_owned());
        }
        if backgrounds.is_empty() {
            return Err("합성에 사용할 배경 이미지가 없습니다.".to_owned());
        }

        let job = visionforge_core::start_job(&project_path, "synthetic_generation", count)
            .map_err(|error| error.to_string())?;
        let output_directory = std::path::Path::new(&project_path)
            .join("assets/generated")
            .join(&job.id);
        let generated = match engine::generate_images(
            &targets,
            &backgrounds,
            &output_directory,
            count,
            seed,
            &task_spec.generation_policy,
        ) {
            Ok(items) => items,
            Err(error) => {
                let _ = visionforge_core::fail_job(&project_path, &job.id, "engine_failed", &error);
                return Err(error);
            }
        };
        visionforge_core::record_generation_results(&project_path, &job.id, &generated)
            .map_err(|error| error.to_string())
    })
    .await
    .map_err(|error| format!("합성 작업이 중단됐습니다: {error}"))?
}

#[tauri::command]
async fn review_images(
    project_path: String,
    asset_ids: Vec<String>,
    review_status: String,
) -> Result<ReviewUpdate, String> {
    tauri::async_runtime::spawn_blocking(move || {
        visionforge_core::set_image_review_status(&project_path, &asset_ids, &review_status)
            .map_err(|error| error.to_string())
    })
    .await
    .map_err(|error| format!("검토 상태 저장 작업이 중단됐습니다: {error}"))?
}

#[tauri::command]
async fn update_bounding_box(
    project_path: String,
    asset_id: String,
    bounding_box: visionforge_core::BoundingBoxInput,
) -> Result<visionforge_core::BoundingBoxInput, String> {
    tauri::async_runtime::spawn_blocking(move || {
        visionforge_core::update_bounding_box(&project_path, &asset_id, &bounding_box)
            .map_err(|error| error.to_string())
    })
    .await
    .map_err(|error| format!("Bounding Box 저장 작업이 중단됐습니다: {error}"))?
}

#[tauri::command]
async fn create_dataset_version(
    project_path: String,
    seed: i64,
) -> Result<DatasetVersionSummary, String> {
    tauri::async_runtime::spawn_blocking(move || {
        visionforge_core::create_dataset_version(&project_path, seed)
            .map_err(|error| error.to_string())
    })
    .await
    .map_err(|error| format!("데이터셋 생성 작업이 중단됐습니다: {error}"))?
}

#[tauri::command]
async fn train_local_model(
    project_path: String,
    dataset_id: String,
    seed: i64,
) -> Result<ModelVersionSummary, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let dataset_path = visionforge_core::dataset_manifest_path(&project_path, &dataset_id)
            .map_err(|error| error.to_string())?;
        let job = visionforge_core::start_job(&project_path, "model_training", 1)
            .map_err(|error| error.to_string())?;
        let safe_dataset_id = dataset_id
            .chars()
            .filter(char::is_ascii_alphanumeric)
            .take(32)
            .collect::<String>();
        if safe_dataset_id.len() < 8 {
            return Err("학습 데이터셋 ID가 올바르지 않습니다.".to_owned());
        }
        let output_directory = std::path::Path::new(&project_path)
            .join("models/.training")
            .join(format!("{safe_dataset_id}-{seed}"));
        let mut result = match engine::train_model(&dataset_path, &output_directory, seed) {
            Ok(result) => result,
            Err(error) => {
                let _ =
                    visionforge_core::fail_job(&project_path, &job.id, "training_failed", &error);
                return Err(error);
            }
        };
        let model_id = result
            .model_id
            .as_deref()
            .ok_or_else(|| "학습 모델 ID가 누락됐습니다.".to_owned())?;
        let final_directory = std::path::Path::new(&project_path)
            .join("models")
            .join(model_id);
        if final_directory.exists() {
            return Err("같은 ID의 모델 폴더가 이미 존재합니다.".to_owned());
        }
        std::fs::rename(&output_directory, &final_directory)
            .map_err(|error| format!("완료 모델 폴더를 확정하지 못했습니다: {error}"))?;
        result.model_path = final_directory
            .join("model.json")
            .to_string_lossy()
            .into_owned();
        result.metrics_path = final_directory
            .join("metrics.json")
            .to_string_lossy()
            .into_owned();
        visionforge_core::record_training_result(&project_path, &job.id, &dataset_id, &result)
            .map_err(|error| error.to_string())
    })
    .await
    .map_err(|error| format!("모델 학습 작업이 중단됐습니다: {error}"))?
}

#[tauri::command]
async fn run_batch_inference(
    project_path: String,
    model_id: String,
    paths: Vec<String>,
) -> Result<InferenceBatchResult, String> {
    tauri::async_runtime::spawn_blocking(move || {
        if paths.is_empty() || paths.len() > 10_000 {
            return Err("한 작업의 추론 이미지는 1개 이상 10,000개 이하여야 합니다.".to_owned());
        }
        let task_spec = visionforge_core::current_task_spec(&project_path)
            .map_err(|error| error.to_string())?;
        let positive_threshold = task_spec.output_policy.positive_threshold;
        let negative_threshold = task_spec.output_policy.negative_threshold;
        let model_path = visionforge_core::model_file_path(&project_path, &model_id)
            .map_err(|error| error.to_string())?;
        let inspections = engine::inspect_images(&paths)?;
        let imported = visionforge_core::import_inference_images(&project_path, &inspections)
            .map_err(|error| error.to_string())?;
        let job =
            visionforge_core::start_job(&project_path, "batch_inference", imported.len() as u64)
                .map_err(|error| error.to_string())?;
        let output_directory = std::path::Path::new(&project_path)
            .join("assets/results")
            .join(&job.id);

        let mut engine_paths = Vec::new();
        let mut results = Vec::new();
        for item in imported {
            if item.status == "succeeded" {
                if let Some(relative) = item.internal_path {
                    engine_paths.push(
                        std::path::Path::new(&project_path)
                            .join(relative)
                            .to_string_lossy()
                            .into_owned(),
                    );
                }
            } else {
                results.push(InferenceImageInput {
                    status: "failed".to_owned(),
                    input_path: item.original_path,
                    output_path: String::new(),
                    detections: Vec::new(),
                    max_confidence: None,
                    checksum_sha256: None,
                    width: None,
                    height: None,
                    elapsed_ms: None,
                    error_code: item.error_code,
                    error_message: item.error_message,
                });
            }
        }
        if !engine_paths.is_empty() {
            match engine::infer_images(
                &model_path,
                &engine_paths,
                &output_directory,
                negative_threshold,
            ) {
                Ok(mut engine_results) => results.append(&mut engine_results),
                Err(error) => {
                    let _ = visionforge_core::fail_job(
                        &project_path,
                        &job.id,
                        "inference_failed",
                        &error,
                    );
                    return Err(error);
                }
            }
        }
        let mut batch = visionforge_core::record_inference_results(
            &project_path,
            &job.id,
            &model_id,
            positive_threshold,
            &task_spec,
            &results,
        )
        .map_err(|error| error.to_string())?;
        visionforge_core::route_inference_results(&project_path, &mut batch, &task_spec)
            .map_err(|error| error.to_string())?;
        Ok(batch)
    })
    .await
    .map_err(|error| format!("일괄 추론 작업이 중단됐습니다: {error}"))?
}

#[tauri::command]
async fn export_model_package(
    project_path: String,
    model_id: String,
    package_path: String,
) -> Result<ModelPackageResultInput, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let (model_path, metrics_path) =
            visionforge_core::model_package_sources(&project_path, &model_id)
                .map_err(|error| error.to_string())?;
        let task_spec = visionforge_core::current_task_spec(&project_path)
            .map_err(|error| error.to_string())?;
        engine::export_model_package(
            &model_path,
            &metrics_path,
            std::path::Path::new(&package_path),
            &task_spec,
        )
    })
    .await
    .map_err(|error| format!("모델 패키지 내보내기 작업이 중단됐습니다: {error}"))?
}

#[tauri::command]
async fn import_model_as_project(
    base_directory: String,
    package_path: String,
) -> Result<ImportedModelProject, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let inspected = engine::import_model_package(std::path::Path::new(&package_path), None)?;
        let class_name = inspected
            .class_name
            .clone()
            .ok_or_else(|| "모델 패키지 클래스 이름이 누락됐습니다.".to_owned())?;
        let project = visionforge_core::create_project(
            &base_directory,
            &format!("{class_name} 추론 프로젝트"),
            &class_name,
        )
        .map_err(|error| error.to_string())?;
        let package_id = inspected
            .package_id
            .as_deref()
            .ok_or_else(|| "모델 패키지 ID가 누락됐습니다.".to_owned())?;
        let safe_id = package_id
            .chars()
            .filter(char::is_ascii_alphanumeric)
            .take(32)
            .collect::<String>();
        if safe_id.len() < 8 {
            return Err("모델 패키지 ID가 올바르지 않습니다.".to_owned());
        }
        let import_root = std::path::Path::new(&project.path)
            .join("models/imported")
            .join(safe_id);
        std::fs::create_dir_all(&import_root)
            .map_err(|error| format!("모델 가져오기 폴더를 만들지 못했습니다: {error}"))?;
        let stored_package = import_root.join("source.vfmodel");
        std::fs::copy(&package_path, &stored_package)
            .map_err(|error| format!("모델 패키지를 프로젝트에 복사하지 못했습니다: {error}"))?;
        let extracted = import_root.join("content");
        let imported = engine::import_model_package(&stored_package, Some(&extracted))?;
        if let Some(task_spec_path) = imported.task_spec_path.as_deref() {
            let packaged: TaskSpec = serde_json::from_slice(
                &std::fs::read(task_spec_path)
                    .map_err(|error| format!("작업 명세 파일을 읽지 못했습니다: {error}"))?,
            )
            .map_err(|error| format!("작업 명세 형식이 올바르지 않습니다: {error}"))?;
            let input = TaskSpecInput {
                task_type: packaged.task_type,
                scenario_description: packaged.scenario_description,
                output_policy: packaged.output_policy,
            };
            visionforge_core::save_task_spec(&project.path, &input)
                .map_err(|error| error.to_string())?;
        }
        let model = visionforge_core::record_imported_model(&project.path, &imported)
            .map_err(|error| error.to_string())?;
        Ok(ImportedModelProject { project, model })
    })
    .await
    .map_err(|error| format!("모델 패키지 가져오기 작업이 중단됐습니다: {error}"))?
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            let filename = if cfg!(target_os = "windows") {
                "visionforge-engine.exe"
            } else {
                "visionforge-engine"
            };
            let engine_path = app
                .path()
                .resource_dir()?
                .join("visionforge-engine-runtime")
                .join(filename);
            engine::configure_bundled_engine(engine_path);
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            system_status,
            create_project,
            open_project_workspace,
            save_task_spec,
            import_target_images,
            import_background_images,
            generate_synthetic_batch,
            review_images,
            update_bounding_box,
            create_dataset_version,
            train_local_model,
            run_batch_inference,
            export_model_package,
            import_model_as_project
        ])
        .run(tauri::generate_context!())
        .expect("VisionForge desktop application failed");
}
