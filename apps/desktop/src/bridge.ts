import { invoke } from "@tauri-apps/api/core";
import { open, save } from "@tauri-apps/plugin-dialog";

import type {
  BoundingBox,
  DatasetVersionSummary,
  GenerationBatchResult,
  ImportedImage,
  InferenceBatchResult,
  ImportedModelProject,
  ModelPackageResult,
  ModelVersionSummary,
  ProjectSummary,
  ProjectWorkspace,
  ReviewUpdate,
  SystemStatus,
  TaskSpec,
  TaskSpecInput,
} from "./types";

export function isDesktopRuntime() {
  return "__TAURI_INTERNALS__" in window;
}

export async function getSystemStatus(): Promise<SystemStatus> {
  if (!isDesktopRuntime()) {
    return {
      offline: true,
      engineReady: false,
      enginePath: "브라우저 미리보기",
      platform: "web-preview",
      hardware: null,
    };
  }
  return invoke<SystemStatus>("system_status");
}

export async function chooseProjectDirectory() {
  if (!isDesktopRuntime()) return null;
  const selected = await open({
    directory: true,
    multiple: false,
    title: "새 VisionForge 프로젝트의 상위 저장 폴더 선택",
  });
  return typeof selected === "string" ? selected : null;
}

export async function chooseExistingProjectDirectory() {
  if (!isDesktopRuntime()) return null;
  const selected = await open({
    directory: true,
    multiple: false,
    title: "project.json이 있는 VisionForge 프로젝트 폴더 선택",
  });
  return typeof selected === "string" ? selected : null;
}

export async function chooseImages() {
  if (!isDesktopRuntime()) return [];
  const selected = await open({
    directory: false,
    multiple: true,
    title: "대상 이미지 여러 장 선택",
    filters: [
      {
        name: "지원 이미지",
        extensions: ["jpg", "jpeg", "png", "webp", "bmp"],
      },
    ],
  });
  if (typeof selected === "string") return [selected];
  return selected ?? [];
}

export async function chooseBackgroundImages() {
  if (!isDesktopRuntime()) return [];
  const selected = await open({
    directory: false,
    multiple: true,
    title: "합성 배경 이미지 여러 장 선택",
    filters: [
      {
        name: "지원 이미지",
        extensions: ["jpg", "jpeg", "png", "webp", "bmp"],
      },
    ],
  });
  if (typeof selected === "string") return [selected];
  return selected ?? [];
}

export async function chooseInferenceImages() {
  if (!isDesktopRuntime()) return [];
  const selected = await open({
    directory: false,
    multiple: true,
    title: "일괄 판정할 실제 이미지 여러 장 선택",
    filters: [
      {
        name: "지원 이미지",
        extensions: ["jpg", "jpeg", "png", "webp", "bmp"],
      },
    ],
  });
  if (typeof selected === "string") return [selected];
  return selected ?? [];
}

export async function chooseModelPackage() {
  if (!isDesktopRuntime()) return null;
  const selected = await open({
    directory: false,
    multiple: false,
    title: "VisionForge 모델 패키지 가져오기",
    filters: [{ name: "VisionForge 모델", extensions: ["vfmodel"] }],
  });
  return typeof selected === "string" ? selected : null;
}

export async function chooseModelExportPath(defaultName: string) {
  if (!isDesktopRuntime()) return null;
  return save({
    title: "VisionForge 모델 패키지 내보내기",
    defaultPath: `${defaultName}.vfmodel`,
    filters: [{ name: "VisionForge 모델", extensions: ["vfmodel"] }],
  });
}

export function createProject(baseDirectory: string, name: string, className: string) {
  return invoke<ProjectSummary>("create_project", { baseDirectory, name, className });
}

export function openProjectWorkspace(projectPath: string) {
  return invoke<ProjectWorkspace>("open_project_workspace", { projectPath });
}

export function saveTaskSpec(projectPath: string, input: TaskSpecInput) {
  return invoke<TaskSpec>("save_task_spec", { projectPath, input });
}

export function importTargetImages(projectPath: string, paths: string[]) {
  return invoke<ImportedImage[]>("import_target_images", { projectPath, paths });
}

export function importBackgroundImages(projectPath: string, paths: string[]) {
  return invoke<ImportedImage[]>("import_background_images", { projectPath, paths });
}

export function generateSyntheticBatch(projectPath: string, count: number, seed: number) {
  return invoke<GenerationBatchResult>("generate_synthetic_batch", {
    projectPath,
    count,
    seed,
  });
}

export function reviewImages(
  projectPath: string,
  assetIds: string[],
  reviewStatus: "approved" | "excluded" | "unreviewed" | "needs_review",
) {
  return invoke<ReviewUpdate>("review_images", { projectPath, assetIds, reviewStatus });
}

export function updateBoundingBox(
  projectPath: string,
  assetId: string,
  boundingBox: BoundingBox,
) {
  return invoke<BoundingBox>("update_bounding_box", { projectPath, assetId, boundingBox });
}

export function createDatasetVersion(projectPath: string, seed: number) {
  return invoke<DatasetVersionSummary>("create_dataset_version", { projectPath, seed });
}

export function trainLocalModel(projectPath: string, datasetId: string, seed: number) {
  return invoke<ModelVersionSummary>("train_local_model", { projectPath, datasetId, seed });
}

export function runBatchInference(
  projectPath: string,
  modelId: string,
  paths: string[],
) {
  return invoke<InferenceBatchResult>("run_batch_inference", {
    projectPath,
    modelId,
    paths,
  });
}

export function exportModelPackage(
  projectPath: string,
  modelId: string,
  packagePath: string,
) {
  return invoke<ModelPackageResult>("export_model_package", {
    projectPath,
    modelId,
    packagePath,
  });
}

export function importModelAsProject(baseDirectory: string, packagePath: string) {
  return invoke<ImportedModelProject>("import_model_as_project", {
    baseDirectory,
    packagePath,
  });
}
