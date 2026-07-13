// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("./bridge", () => ({
  chooseBackgroundImages: vi.fn(),
  chooseExistingProjectDirectory: vi.fn(),
  chooseImages: vi.fn(),
  chooseInferenceImages: vi.fn(),
  chooseModelExportPath: vi.fn(),
  chooseModelPackage: vi.fn(),
  chooseProjectDirectory: vi.fn(),
  createDatasetVersion: vi.fn(),
  createProject: vi.fn(),
  exportModelPackage: vi.fn(),
  generateSyntheticBatch: vi.fn(),
  getSystemStatus: vi.fn(),
  importBackgroundImages: vi.fn(),
  importModelAsProject: vi.fn(),
  importTargetImages: vi.fn(),
  openProjectWorkspace: vi.fn(),
  reviewImages: vi.fn(),
  runBatchInference: vi.fn(),
  saveTaskSpec: vi.fn(),
  trainLocalModel: vi.fn(),
  updateBoundingBox: vi.fn(),
}));

import App from "./App";
import {
  chooseExistingProjectDirectory,
  getSystemStatus,
  openProjectWorkspace,
} from "./bridge";
import type { ProjectWorkspace, SystemStatus } from "./types";

const systemStatus: SystemStatus = {
  offline: true,
  engineReady: true,
  enginePath: "/Applications/VisionForge.app/engine",
  platform: "darwin",
  hardware: null,
};

const workspace: ProjectWorkspace = {
  project: {
    id: "project-1",
    name: "기존 검사 프로젝트",
    classId: "class-1",
    className: "빨간 부품",
    path: "/Users/test/VisionForge/vf-project1",
    createdAt: "2026-07-13T00:00:00Z",
    imageCount: 0,
    warningCount: 0,
  },
  targets: [],
  backgrounds: [],
  generated: [],
  dataset: null,
  model: null,
  taskSpec: {
    schemaVersion: 1,
    id: "task-1",
    revision: 1,
    taskType: "object_presence",
    pipelineId: "torchvision_fasterrcnn_v1",
    classId: "class-1",
    className: "빨간 부품",
    scenarioDescription: "",
    compiledTags: [],
    generationPolicy: {
      scaleMin: 0.2,
      scaleMax: 0.8,
      rotationMin: -15,
      rotationMax: 15,
      brightnessMin: 0.8,
      brightnessMax: 1.2,
      contrastMin: 0.8,
      contrastMax: 1.2,
      blurRadiusMax: 0,
      noiseStddevMax: 0,
      occlusionMax: 0,
    },
    outputPolicy: {
      presentFolder: "대상_포함",
      absentFolder: "대상_미포함",
      reviewFolder: "검토_필요",
      failedFolder: "처리_실패",
      positiveThreshold: 0.85,
      negativeThreshold: 0.35,
      copyMode: "copy_original",
    },
    compiler: "visionforge_rules_v1",
    warnings: [],
    createdAt: "2026-07-13T00:00:00Z",
  },
  recoveredJobs: 0,
};

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(getSystemStatus).mockResolvedValue(systemStatus);
});

afterEach(() => cleanup());

describe("project selection navigation", () => {
  it("returns to project selection without discarding the active workspace", async () => {
    vi.mocked(chooseExistingProjectDirectory).mockResolvedValue(workspace.project.path);
    vi.mocked(openProjectWorkspace).mockResolvedValue(workspace);
    render(<App />);

    fireEvent.click(screen.getByRole("button", { name: "저장된 VisionForge 프로젝트 열기" }));
    expect(await screen.findByRole("heading", { name: workspace.project.name })).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: "프로젝트 선택 화면" }));
    expect(screen.getByRole("button", { name: "현재 프로젝트로 돌아가기" })).toBeTruthy();
    const nameInput = screen.getByLabelText("프로젝트 이름") as HTMLInputElement;
    expect(nameInput.value).toBe(workspace.project.name);

    fireEvent.change(nameInput, { target: { value: "새 프로젝트 이름" } });
    expect(nameInput.value).toBe("새 프로젝트 이름");
    fireEvent.click(screen.getByRole("button", { name: "현재 프로젝트로 돌아가기" }));

    expect(screen.getByRole("heading", { name: workspace.project.name })).toBeTruthy();
    expect(screen.getByText(workspace.project.path)).toBeTruthy();
  });

  it("keeps the setup screen usable when an empty directory is rejected", async () => {
    vi.mocked(chooseExistingProjectDirectory).mockResolvedValue("/Users/test/empty");
    vi.mocked(openProjectWorkspace).mockRejectedValue(
      "선택한 폴더는 VisionForge 프로젝트가 아닙니다. 새 프로젝트를 만들어 주세요.",
    );
    render(<App />);

    const openButton = screen.getByRole("button", { name: "저장된 VisionForge 프로젝트 열기" });
    fireEvent.click(openButton);

    const notice = await screen.findByRole("status");
    expect(notice.textContent).toContain("VisionForge 프로젝트가 아닙니다");
    expect(screen.getByRole("heading", { name: "첫 탐지 대상을 정의합니다." })).toBeTruthy();
    expect((openButton as HTMLButtonElement).disabled).toBe(false);
  });
});
