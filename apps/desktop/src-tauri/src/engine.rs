use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::OnceLock;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use visionforge_core::{
    GeneratedImageInput, GenerationPolicySpec, ImageInspectionInput, InferenceImageInput,
    ModelPackageResultInput, TaskSpec, TrainingResultInput,
};

static BUNDLED_ENGINE: OnceLock<PathBuf> = OnceLock::new();

#[derive(Debug, Deserialize)]
struct EngineEnvelope {
    status: String,
    #[serde(default)]
    items: Vec<ImageInspectionInput>,
    error_message: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GenerationEnvelope {
    status: String,
    #[serde(default)]
    items: Vec<GeneratedImageInput>,
    error_message: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TrainingEnvelope {
    status: String,
    result: Option<TrainingResultInput>,
    error_message: Option<String>,
}

#[derive(Debug, Deserialize)]
struct InferenceEnvelope {
    status: String,
    #[serde(default)]
    items: Vec<InferenceImageInput>,
    error_message: Option<String>,
}

#[derive(Debug, Deserialize)]
struct PackageEnvelope {
    status: String,
    result: Option<ModelPackageResultInput>,
    error_message: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all(serialize = "camelCase", deserialize = "snake_case"))]
pub struct HardwareProfile {
    pub profile: String,
    pub platform: String,
    pub architecture: String,
    pub cpu_count: u64,
    pub total_memory_bytes: Option<u64>,
    pub accelerator: String,
    pub accelerator_name: String,
    pub accelerator_memory_bytes: Option<u64>,
    pub execution_providers: Vec<String>,
    pub free_disk_bytes: Option<u64>,
    pub torch_version: Option<String>,
    pub torchvision_version: Option<String>,
}

#[derive(Debug, Deserialize)]
struct HardwareProfileEnvelope {
    status: String,
    result: Option<HardwareProfile>,
    error_message: Option<String>,
}

pub fn engine_ready() -> bool {
    bundled_engine_path().is_some() || (cfg!(debug_assertions) && python_path().exists())
}

pub fn engine_display_path() -> String {
    if let Some(path) = bundled_engine_path() {
        path.to_string_lossy().into_owned()
    } else if cfg!(debug_assertions) {
        python_path().to_string_lossy().into_owned()
    } else {
        "번들 이미지 엔진을 찾을 수 없습니다.".to_owned()
    }
}

pub fn configure_bundled_engine(path: PathBuf) {
    if path.is_file() {
        let _ = BUNDLED_ENGINE.set(path);
    }
}

pub fn hardware_profile(path: Option<&Path>) -> Result<HardwareProfile, String> {
    let request = serde_json::json!({ "path": path });
    let output = run_engine_command("system-profile", &request, "장치 프로필 확인")?;
    let response: HardwareProfileEnvelope = serde_json::from_slice(&output)
        .map_err(|error| format!("장치 프로필 응답 형식이 올바르지 않습니다: {error}"))?;
    if response.status != "succeeded" {
        return Err(response
            .error_message
            .unwrap_or_else(|| "장치 프로필 확인에 실패했습니다.".to_owned()));
    }
    response
        .result
        .ok_or_else(|| "장치 프로필 결과가 누락됐습니다.".to_owned())
}

fn bundled_engine_path() -> Option<PathBuf> {
    if let Some(path) = env::var_os("VISIONFORGE_ENGINE_EXECUTABLE") {
        let path = PathBuf::from(path);
        return path.exists().then_some(path);
    }
    if let Some(path) = BUNDLED_ENGINE.get() {
        return path.exists().then(|| path.clone());
    }
    if cfg!(debug_assertions) {
        return None;
    }
    let executable = env::current_exe().ok()?;
    let executable_directory = executable.parent()?;
    let base_directory = if executable_directory.ends_with("deps") {
        executable_directory
            .parent()
            .unwrap_or(executable_directory)
    } else {
        executable_directory
    };
    let filename = if cfg!(target_os = "windows") {
        "visionforge-engine.exe"
    } else {
        "visionforge-engine"
    };
    let path = base_directory.join(filename);
    path.exists().then_some(path)
}

pub fn engine_project_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../engine")
}

pub fn python_path() -> PathBuf {
    if let Some(path) = env::var_os("VISIONFORGE_PYTHON") {
        return PathBuf::from(path);
    }
    let engine = engine_project_path();
    if cfg!(target_os = "windows") {
        engine.join(".venv/Scripts/python.exe")
    } else {
        engine.join(".venv/bin/python")
    }
}

pub fn inspect_images(paths: &[String]) -> Result<Vec<ImageInspectionInput>, String> {
    let request = serde_json::json!({ "paths": paths });
    let output = run_engine_command("inspect", &request, "이미지 검사")?;
    let response: EngineEnvelope = serde_json::from_slice(&output)
        .map_err(|error| format!("이미지 엔진 응답 형식이 올바르지 않습니다: {error}"))?;
    if response.status != "succeeded" {
        return Err(response
            .error_message
            .unwrap_or_else(|| "이미지 엔진 작업이 실패했습니다.".to_owned()));
    }
    Ok(response.items)
}

pub fn generate_images(
    target_paths: &[String],
    background_paths: &[String],
    output_directory: &Path,
    count: u64,
    seed: i64,
    policy: &GenerationPolicySpec,
) -> Result<Vec<GeneratedImageInput>, String> {
    let request = serde_json::json!({
        "target_paths": target_paths,
        "background_paths": background_paths,
        "output_directory": output_directory,
        "count": count,
        "seed": seed,
        "policy": {
            "scale_min": policy.scale_min,
            "scale_max": policy.scale_max,
            "rotation_min": policy.rotation_min,
            "rotation_max": policy.rotation_max,
            "brightness_min": policy.brightness_min,
            "brightness_max": policy.brightness_max,
            "contrast_min": policy.contrast_min,
            "contrast_max": policy.contrast_max,
            "blur_radius_max": policy.blur_radius_max,
            "noise_stddev_max": policy.noise_stddev_max,
            "occlusion_max": policy.occlusion_max,
        },
    });
    let output = run_engine_command("generate", &request, "이미지 합성")?;
    let response: GenerationEnvelope = serde_json::from_slice(&output)
        .map_err(|error| format!("합성 엔진 응답 형식이 올바르지 않습니다: {error}"))?;
    if response.status != "succeeded" {
        return Err(response
            .error_message
            .unwrap_or_else(|| "이미지 합성 작업이 실패했습니다.".to_owned()));
    }
    Ok(response.items)
}

fn run_engine_command(
    command: &str,
    request: &serde_json::Value,
    operation: &str,
) -> Result<Vec<u8>, String> {
    if let Some(engine) = bundled_engine_path() {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| format!("{operation} 요청 시각을 만들 수 없습니다: {error}"))?
            .as_nanos();
        let request_path = env::temp_dir().join(format!(
            "visionforge-engine-{}-{timestamp}.json",
            std::process::id()
        ));
        fs::write(&request_path, request.to_string())
            .map_err(|error| format!("{operation} 임시 요청 파일을 쓰지 못했습니다: {error}"))?;
        let request_argument = request_path.to_string_lossy().into_owned();
        let result = (|| {
            let mut sidecar = Command::new(engine);
            sidecar.args([command, "--input", &request_argument]);
            #[cfg(target_os = "windows")]
            {
                use std::os::windows::process::CommandExt;
                sidecar.creation_flags(0x0800_0000);
            }
            let output = sidecar
                .output()
                .map_err(|error| format!("{operation} sidecar 실행에 실패했습니다: {error}"))?;
            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                return Err(format!("{operation} sidecar가 실패했습니다: {stderr}"));
            }
            Ok(output.stdout)
        })();
        let _ = fs::remove_file(request_path);
        return result;
    }

    if !cfg!(debug_assertions) {
        return Err(
            "배포 앱에 포함된 로컬 이미지 엔진을 찾을 수 없습니다. 앱을 다시 설치해 주세요."
                .to_owned(),
        );
    }

    let python = python_path();
    if !python.exists() {
        return Err(format!(
            "로컬 이미지 엔진을 찾을 수 없습니다: {}",
            python.display()
        ));
    }
    let mut child = Command::new(&python)
        .current_dir(engine_project_path())
        .args(["-m", "visionforge_engine", command, "--input", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("{operation} 엔진 실행에 실패했습니다: {error}"))?;
    child
        .stdin
        .take()
        .ok_or_else(|| format!("{operation} 엔진 입력 스트림을 열 수 없습니다."))?
        .write_all(request.to_string().as_bytes())
        .map_err(|error| format!("{operation} 요청 전달에 실패했습니다: {error}"))?;
    let output = child
        .wait_with_output()
        .map_err(|error| format!("{operation} 엔진 응답을 기다리지 못했습니다: {error}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("{operation} 엔진이 실패했습니다: {stderr}"));
    }
    Ok(output.stdout)
}

pub fn train_model(
    dataset_manifest_path: &Path,
    output_directory: &Path,
    seed: i64,
) -> Result<TrainingResultInput, String> {
    train_model_with_backend(dataset_manifest_path, output_directory, seed, "auto")
}

fn train_model_with_backend(
    dataset_manifest_path: &Path,
    output_directory: &Path,
    seed: i64,
    backend: &str,
) -> Result<TrainingResultInput, String> {
    let request = serde_json::json!({
        "dataset_manifest_path": dataset_manifest_path,
        "output_directory": output_directory,
        "seed": seed,
        "backend": backend,
    });
    let output = run_engine_command("train", &request, "학습")?;
    let response: TrainingEnvelope = serde_json::from_slice(&output)
        .map_err(|error| format!("학습 엔진 응답 형식이 올바르지 않습니다: {error}"))?;
    let TrainingEnvelope {
        status,
        result,
        error_message,
    } = response;
    let result = result.ok_or_else(|| {
        error_message
            .clone()
            .unwrap_or_else(|| "학습 엔진 결과가 누락됐습니다.".to_owned())
    })?;
    if status != "succeeded" || result.status != "succeeded" {
        return Err(result
            .error_message
            .clone()
            .or(error_message)
            .unwrap_or_else(|| "학습 작업이 실패했습니다.".to_owned()));
    }
    Ok(result)
}

pub fn infer_images(
    model_path: &Path,
    input_paths: &[String],
    output_directory: &Path,
    confidence_threshold: f64,
) -> Result<Vec<InferenceImageInput>, String> {
    let request = serde_json::json!({
        "model_path": model_path,
        "input_paths": input_paths,
        "output_directory": output_directory,
        "confidence_threshold": confidence_threshold,
    });
    let output = run_engine_command("infer", &request, "추론")?;
    let response: InferenceEnvelope = serde_json::from_slice(&output)
        .map_err(|error| format!("추론 엔진 응답 형식이 올바르지 않습니다: {error}"))?;
    if response.status != "succeeded" {
        return Err(response
            .error_message
            .unwrap_or_else(|| "추론 작업이 실패했습니다.".to_owned()));
    }
    Ok(response.items)
}

fn package_result(output: &[u8], operation: &str) -> Result<ModelPackageResultInput, String> {
    let response: PackageEnvelope = serde_json::from_slice(output)
        .map_err(|error| format!("{operation} 응답 형식이 올바르지 않습니다: {error}"))?;
    let PackageEnvelope {
        status,
        result,
        error_message,
    } = response;
    let result = result.ok_or_else(|| {
        error_message
            .clone()
            .unwrap_or_else(|| format!("{operation} 결과가 누락됐습니다."))
    })?;
    if status != "succeeded" || result.status != "succeeded" {
        return Err(result
            .error_message
            .clone()
            .or(error_message)
            .unwrap_or_else(|| format!("{operation} 작업이 실패했습니다.")));
    }
    Ok(result)
}

pub fn export_model_package(
    model_path: &Path,
    metrics_path: &Path,
    package_path: &Path,
    task_spec: &TaskSpec,
) -> Result<ModelPackageResultInput, String> {
    let request = serde_json::json!({
        "model_path": model_path,
        "metrics_path": metrics_path,
        "package_path": package_path,
        "app_version": env!("CARGO_PKG_VERSION"),
        "task_spec": task_spec,
    });
    let output = run_engine_command("export-model", &request, "모델 패키지 내보내기")?;
    package_result(&output, "모델 패키지 내보내기")
}

pub fn import_model_package(
    package_path: &Path,
    extract_directory: Option<&Path>,
) -> Result<ModelPackageResultInput, String> {
    let request = serde_json::json!({
        "package_path": package_path,
        "extract_directory": extract_directory,
    });
    let output = run_engine_command("import-model", &request, "모델 패키지 가져오기")?;
    package_result(&output, "모델 패키지 가져오기")
}

#[cfg(test)]
mod tests {
    use std::fs;

    use base64::Engine;

    use super::*;

    #[test]
    fn python_engine_contract_inspects_a_real_png() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let image_path = directory.path().join("pixel.png");
        let png = base64::engine::general_purpose::STANDARD
            .decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=",
            )
            .expect("valid fixture");
        fs::write(&image_path, png).expect("write fixture");

        let result =
            inspect_images(&[image_path.to_string_lossy().into_owned()]).expect("engine response");

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].status, "succeeded");
        assert_eq!(result[0].width, Some(1));
        assert_eq!(result[0].height, Some(1));
        assert!(result[0].checksum_sha256.is_some());
    }

    #[test]
    fn python_engine_contract_generates_a_labeled_image() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let target_path = directory.path().join("target.png");
        let background_path = directory.path().join("background.png");
        let output_directory = directory.path().join("generated");
        let target_png = base64::engine::general_purpose::STANDARD
            .decode(
                "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAAYElEQVR42mNgGAUjHTDik7wZa/mfGpaoLz6O0x4mWltOyCwmWltOyEwmeliOz2ymgU6Eow4YdcCoA0YdMOqAUQeMOmDUAUykNCBp0ThlIrUVS+2WMRM5TWlqNstHwSgAAJVuHDAAGEwUAAAAAElFTkSuQmCC",
            )
            .expect("valid target fixture");
        let background_png = base64::engine::general_purpose::STANDARD
            .decode(
                "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAYAAACqaXHeAAAAZ0lEQVR42u3QMQEAIAgAMKR/Ok8/k0AP2CLsvH8rFstYToAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIETNA4fgQMHhGuagAAAABJRU5ErkJggg==",
            )
            .expect("valid background fixture");
        fs::write(&target_path, target_png).expect("write target fixture");
        fs::write(&background_path, background_png).expect("write background fixture");
        let target = target_path.to_string_lossy().into_owned();
        let background = background_path.to_string_lossy().into_owned();

        let result = generate_images(
            std::slice::from_ref(&target),
            std::slice::from_ref(&background),
            &output_directory,
            1,
            2026,
            &GenerationPolicySpec::default(),
        )
        .expect("generation response");

        assert_eq!(result.len(), 1);
        assert_eq!(
            result[0].status, "succeeded",
            "generation failed: {:?}",
            result[0].error_message
        );
        assert!(result[0].bounding_box.is_some());
        assert!(Path::new(&result[0].output_path).exists());
        assert!(result[0].checksum_sha256.is_some());
    }

    #[test]
    fn full_local_pipeline_reaches_recorded_inference() {
        let root = tempfile::tempdir().expect("temporary directory");
        let project = visionforge_core::create_project(root.path(), "종단간", "빨간 부품")
            .expect("create project");
        let target_path = root.path().join("target.png");
        let background_path = root.path().join("background.png");
        let target_png = base64::engine::general_purpose::STANDARD
            .decode(
                "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAAYElEQVR42mNgGAUjHTDik7wZa/mfGpaoLz6O0x4mWltOyCwmWltOyEwmeliOz2ymgU6Eow4YdcCoA0YdMOqAUQeMOmDUAUykNCBp0ThlIrUVS+2WMRM5TWlqNstHwSgAAJVuHDAAGEwUAAAAAElFTkSuQmCC",
            )
            .expect("target fixture");
        let background_png = base64::engine::general_purpose::STANDARD
            .decode(
                "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAYAAACqaXHeAAAAZ0lEQVR42u3QMQEAIAgAMKR/Ok8/k0AP2CLsvH8rFstYToAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIETNA4fgQMHhGuagAAAABJRU5ErkJggg==",
            )
            .expect("background fixture");
        fs::write(&target_path, target_png).expect("write target");
        fs::write(&background_path, background_png).expect("write background");

        let target_inspection =
            inspect_images(&[target_path.to_string_lossy().into_owned()]).expect("inspect target");
        let background_inspection =
            inspect_images(&[background_path.to_string_lossy().into_owned()])
                .expect("inspect background");
        visionforge_core::import_target_images(&project.path, &target_inspection)
            .expect("import target");
        let backgrounds =
            visionforge_core::import_background_images(&project.path, &background_inspection)
                .expect("import background");

        let generation_job = visionforge_core::start_job(&project.path, "synthetic_generation", 1)
            .expect("generation job");
        let targets = visionforge_core::list_image_paths(&project.path, "target_original")
            .expect("target paths");
        let background_paths = visionforge_core::list_image_paths(&project.path, "background")
            .expect("background paths");
        let generated = generate_images(
            &targets,
            &background_paths,
            &Path::new(&project.path)
                .join("assets/generated")
                .join(&generation_job.id),
            1,
            2026,
            &GenerationPolicySpec::default(),
        )
        .expect("generate");
        let generated = visionforge_core::record_generation_results(
            &project.path,
            &generation_job.id,
            &generated,
        )
        .expect("record generation");
        let approved = vec![
            generated.items[0].id.clone().expect("generated id"),
            backgrounds[0].id.clone(),
        ];
        visionforge_core::set_image_review_status(&project.path, &approved, "approved")
            .expect("approve inputs");

        let dataset =
            visionforge_core::create_dataset_version(&project.path, 2026).expect("dataset");
        let training_job =
            visionforge_core::start_job(&project.path, "model_training", 1).expect("training job");
        // This test verifies the Rust/Python/project contract. The dedicated
        // Python integration test exercises the resource-intensive Torch backend.
        let training = train_model_with_backend(
            Path::new(&dataset.manifest_path),
            &Path::new(&project.path)
                .join("models")
                .join(&training_job.id),
            2026,
            "linear",
        )
        .expect("train model");
        let model = visionforge_core::record_training_result(
            &project.path,
            &training_job.id,
            &dataset.id,
            &training,
        )
        .expect("record model");

        let generated_path = generated.items[0].output_path.clone();
        let inference_inspection =
            inspect_images(std::slice::from_ref(&generated_path)).expect("inspect inference input");
        let inference_inputs =
            visionforge_core::import_inference_images(&project.path, &inference_inspection)
                .expect("import inference input");
        let input_path = Path::new(&project.path).join(
            inference_inputs[0]
                .internal_path
                .as_deref()
                .expect("inference internal path"),
        );
        let inference_job = visionforge_core::start_job(&project.path, "batch_inference", 1)
            .expect("inference job");
        let inference = infer_images(
            Path::new(&model.model_path),
            &[input_path.to_string_lossy().into_owned()],
            &Path::new(&project.path)
                .join("assets/results")
                .join(&inference_job.id),
            0.1,
        )
        .expect("infer");
        let task_spec = visionforge_core::current_task_spec(&project.path).expect("task spec");
        let mut recorded = visionforge_core::record_inference_results(
            &project.path,
            &inference_job.id,
            &model.id,
            task_spec.output_policy.positive_threshold,
            &task_spec,
            &inference,
        )
        .expect("record inference");
        visionforge_core::route_inference_results(&project.path, &mut recorded, &task_spec)
            .expect("route inference");

        assert_eq!(recorded.job.status, "succeeded");
        assert_eq!(recorded.items.len(), 1);
        assert!(recorded.items[0].output_path.is_some());
        assert!(recorded.items[0].routed_path.is_some());
        assert!(
            recorded
                .routing
                .as_ref()
                .is_some_and(|value| { Path::new(&value.manifest_path).exists() })
        );

        let (model_source, metrics_source) =
            visionforge_core::model_package_sources(&project.path, &model.id)
                .expect("package sources");
        let package_path = root.path().join("shared.vfmodel");
        let package =
            export_model_package(&model_source, &metrics_source, &package_path, &task_spec)
                .expect("export package");
        assert!(package.package_checksum_sha256.is_some());

        let imported_project =
            visionforge_core::create_project(root.path(), "가져온 모델", "빨간 부품")
                .expect("import project");
        let import_root = Path::new(&imported_project.path).join("models/imported/test-package");
        fs::create_dir_all(&import_root).expect("import directory");
        let stored_package = import_root.join("source.vfmodel");
        fs::copy(&package_path, &stored_package).expect("copy package");
        let imported = import_model_package(&stored_package, Some(&import_root.join("content")))
            .expect("import package");
        assert!(
            imported
                .task_spec_path
                .as_deref()
                .is_some_and(|value| Path::new(value).exists())
        );
        let imported_model =
            visionforge_core::record_imported_model(&imported_project.path, &imported)
                .expect("record imported model");

        assert_eq!(imported_model.origin, "imported");
        assert_eq!(imported_model.class_name, "빨간 부품");
        assert!(Path::new(&imported_model.model_path).exists());
    }
}
