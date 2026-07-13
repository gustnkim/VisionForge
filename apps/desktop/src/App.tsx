import { useEffect, useState } from "react";

import {
  chooseBackgroundImages,
  chooseExistingProjectDirectory,
  chooseImages,
  chooseInferenceImages,
  chooseModelExportPath,
  chooseModelPackage,
  chooseProjectDirectory,
  createDatasetVersion,
  createProject,
  exportModelPackage,
  generateSyntheticBatch,
  getSystemStatus,
  importBackgroundImages,
  importModelAsProject,
  importTargetImages,
  openProjectWorkspace,
  reviewImages,
  runBatchInference,
  saveTaskSpec,
  trainLocalModel,
  updateBoundingBox,
} from "./bridge";
import { compactChecksum, inspectionSummary } from "./domain";
import type {
  BoundingBox,
  DatasetVersionSummary,
  GeneratedImage,
  ImportedImage,
  InferenceBatchResult,
  ModelVersionSummary,
  ProjectSummary,
  ProjectWorkspace,
  SystemStatus,
  TaskSpec,
  TaskSpecInput,
} from "./types";

const workflow = [
  ["01", "대상 등록"],
  ["02", "상황 설계"],
  ["03", "합성 검토"],
  ["04", "데이터셋"],
  ["05", "모델 학습"],
  ["06", "일괄 추론"],
] as const;

type AppView = "setup" | "workspace";

function decisionLabel(decision: string) {
  return {
    present: "포함",
    absent: "미포함",
    review: "검토 필요",
    failed: "처리 실패",
    unrouted: "분류 대기",
  }[decision] ?? decision;
}

function BoundingBoxEditor({
  image,
  busy,
  onSave,
}: {
  image: GeneratedImage;
  busy: boolean;
  onSave: (boundingBox: BoundingBox) => Promise<void>;
}) {
  const [value, setValue] = useState<BoundingBox | null>(image.boundingBox);

  useEffect(() => setValue(image.boundingBox), [image.boundingBox]);
  if (!value) return null;

  return (
    <details className="box-editor">
      <summary>Box 좌표 수정</summary>
      <div>
        {(["xMin", "yMin", "xMax", "yMax"] as const).map((field) => (
          <label key={field}>
            {field}
            <input
              type="number"
              min="0"
              value={value[field]}
              onChange={(event) =>
                setValue((current) =>
                  current ? { ...current, [field]: Number(event.target.value) } : current,
                )
              }
            />
          </label>
        ))}
      </div>
      <button onClick={() => onSave(value)} disabled={busy}>수정 좌표 저장</button>
    </details>
  );
}

function App() {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [project, setProject] = useState<ProjectSummary | null>(null);
  const [view, setView] = useState<AppView>("setup");
  const [images, setImages] = useState<ImportedImage[]>([]);
  const [backgrounds, setBackgrounds] = useState<ImportedImage[]>([]);
  const [generated, setGenerated] = useState<GeneratedImage[]>([]);
  const [dataset, setDataset] = useState<DatasetVersionSummary | null>(null);
  const [model, setModel] = useState<ModelVersionSummary | null>(null);
  const [inference, setInference] = useState<InferenceBatchResult | null>(null);
  const [taskSpec, setTaskSpec] = useState<TaskSpec | null>(null);
  const [projectName, setProjectName] = useState("첫 번째 검사 프로젝트");
  const [className, setClassName] = useState("");
  const [scenarioDescription, setScenarioDescription] = useState("");
  const [presentFolder, setPresentFolder] = useState("대상_포함");
  const [absentFolder, setAbsentFolder] = useState("대상_미포함");
  const [reviewFolder, setReviewFolder] = useState("검토_필요");
  const [failedFolder, setFailedFolder] = useState("처리_실패");
  const [positiveThreshold, setPositiveThreshold] = useState(0.85);
  const [negativeThreshold, setNegativeThreshold] = useState(0.35);
  const [generationCount, setGenerationCount] = useState(24);
  const [generationSeed, setGenerationSeed] = useState(20260712);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);

  useEffect(() => {
    getSystemStatus().then(setStatus).catch((error) => setNotice(String(error)));
  }, []);

  function applyTaskSpec(spec: TaskSpec) {
    setTaskSpec(spec);
    setScenarioDescription(spec.scenarioDescription);
    setPresentFolder(spec.outputPolicy.presentFolder);
    setAbsentFolder(spec.outputPolicy.absentFolder);
    setReviewFolder(spec.outputPolicy.reviewFolder);
    setFailedFolder(spec.outputPolicy.failedFolder);
    setPositiveThreshold(spec.outputPolicy.positiveThreshold);
    setNegativeThreshold(spec.outputPolicy.negativeThreshold);
  }

  function applyWorkspace(workspace: ProjectWorkspace) {
    setProject(workspace.project);
    setProjectName(workspace.project.name);
    setClassName(workspace.project.className);
    setImages(workspace.targets);
    setBackgrounds(workspace.backgrounds);
    setGenerated(workspace.generated);
    setDataset(workspace.dataset);
    setModel(workspace.model);
    setInference(null);
    applyTaskSpec(workspace.taskSpec);
    setView("workspace");
  }

  function currentTaskInput(): TaskSpecInput {
    return {
      taskType: "object_presence",
      scenarioDescription,
      outputPolicy: {
        presentFolder,
        absentFolder,
        reviewFolder,
        failedFolder,
        positiveThreshold,
        negativeThreshold,
        copyMode: "copy_original",
      },
    };
  }

  function taskFormMatchesSavedSpec() {
    if (!taskSpec) return false;
    const input = currentTaskInput();
    return (
      taskSpec.scenarioDescription === input.scenarioDescription.trim() &&
      taskSpec.outputPolicy.presentFolder === input.outputPolicy.presentFolder.trim() &&
      taskSpec.outputPolicy.absentFolder === input.outputPolicy.absentFolder.trim() &&
      taskSpec.outputPolicy.reviewFolder === input.outputPolicy.reviewFolder.trim() &&
      taskSpec.outputPolicy.failedFolder === input.outputPolicy.failedFolder.trim() &&
      taskSpec.outputPolicy.positiveThreshold === input.outputPolicy.positiveThreshold &&
      taskSpec.outputPolicy.negativeThreshold === input.outputPolicy.negativeThreshold
    );
  }

  function confirmProjectReplacement() {
    if (!project || !taskSpec || taskFormMatchesSavedSpec()) return true;
    return window.confirm(
      "현재 프로젝트에 저장하지 않은 작업 설정이 있습니다. 저장하지 않고 다른 프로젝트로 전환할까요?",
    );
  }

  function handleShowProjectSetup() {
    if (!project || busy) return;
    setProjectName(project.name);
    setClassName(project.className);
    setView("setup");
    setNotice(
      "현재 프로젝트는 그대로 유지됩니다. 새 프로젝트를 만들거나 저장된 VisionForge 프로젝트를 선택할 수 있습니다.",
    );
  }

  function handleResumeProject() {
    if (!project || busy) return;
    setView("workspace");
    setNotice(`현재 프로젝트로 돌아왔습니다: ${project.name}`);
  }

  async function persistTaskConfiguration(showNotice: boolean) {
    if (!project) return null;
    if (taskFormMatchesSavedSpec()) return taskSpec;
    const previousGeneration = taskSpec?.generationPolicy;
    const saved = await saveTaskSpec(project.path, currentTaskInput());
    applyTaskSpec(saved);
    if (
      previousGeneration &&
      JSON.stringify(previousGeneration) !== JSON.stringify(saved.generationPolicy)
    ) {
      setDataset(null);
      setModel(null);
      setInference(null);
    }
    if (showNotice) {
      setNotice(
        `작업 명세 r${saved.revision} 저장 완료: ${saved.compiledTags.length > 0 ? saved.compiledTags.join(", ") : "기본 생성 정책"}`,
      );
    }
    return saved;
  }

  async function handleCreateProject() {
    if (!projectName.trim() || !className.trim()) {
      setNotice("프로젝트 이름과 대상 클래스 이름을 모두 입력해 주세요.");
      return;
    }
    const baseDirectory = await chooseProjectDirectory();
    if (!baseDirectory) {
      setNotice("데스크톱 앱에서 프로젝트 저장 폴더를 선택해 주세요.");
      return;
    }
    if (!confirmProjectReplacement()) return;
    setBusy(true);
    setNotice(null);
    try {
      const created = await createProject(baseDirectory, projectName, className);
      const workspace = await openProjectWorkspace(created.path);
      applyWorkspace(workspace);
    } catch (error) {
      setNotice(String(error));
    } finally {
      setBusy(false);
    }
  }

  async function handleImportModelPackage() {
    const packagePath = await chooseModelPackage();
    if (!packagePath) return;
    const baseDirectory = await chooseProjectDirectory();
    if (!baseDirectory) return;
    if (!confirmProjectReplacement()) return;
    setBusy(true);
    setNotice("패키지 경로·압축 크기·파일별 체크섬과 모델 호환성을 검사하고 있습니다.");
    try {
      const result = await importModelAsProject(baseDirectory, packagePath);
      const workspace = await openProjectWorkspace(result.project.path);
      applyWorkspace(workspace);
      setNotice(
        result.model.deploymentStatus === "qualified"
          ? `${result.model.className} 모델을 검증해 가져왔습니다. 실제 이미지 일괄 판정을 시작할 수 있습니다.`
          : `${result.model.className} 후보 모델을 가져왔습니다. 실제 사진 품질 게이트 전에는 모든 성공 결과를 검토 폴더로 보냅니다.`,
      );
    } catch (error) {
      setNotice(String(error));
    } finally {
      setBusy(false);
    }
  }

  async function handleOpenProject() {
    const projectPath = await chooseExistingProjectDirectory();
    if (!projectPath) return;
    if (project?.path === projectPath) {
      setView("workspace");
      setNotice(`현재 프로젝트로 돌아왔습니다: ${project.name}`);
      return;
    }
    if (!confirmProjectReplacement()) return;
    setBusy(true);
    setNotice("저장된 자산, 승인 상태, 데이터셋과 최신 모델을 복원하고 있습니다.");
    try {
      const workspace = await openProjectWorkspace(projectPath);
      applyWorkspace(workspace);
      setNotice(
        `프로젝트를 복원했습니다: 대상 ${workspace.targets.length}, 합성 ${workspace.generated.length}, ${workspace.model ? "모델 준비됨" : "학습 전"}${workspace.recoveredJobs > 0 ? ` · 중단 작업 ${workspace.recoveredJobs}건 보존` : ""}`,
      );
    } catch (error) {
      setNotice(String(error));
    } finally {
      setBusy(false);
    }
  }

  async function handleImportImages() {
    if (!project) return;
    const paths = await chooseImages();
    if (paths.length === 0) return;
    setBusy(true);
    setNotice(`${paths.length}개 이미지를 로컬에서 검사하고 있습니다.`);
    try {
      const imported = await importTargetImages(project.path, paths);
      setImages((current) => [...current, ...imported]);
      setProject((current) =>
        current
          ? {
              ...current,
              imageCount:
                current.imageCount +
                imported.filter((item) => item.status === "succeeded" && !item.duplicate).length,
              warningCount:
                current.warningCount +
                imported.filter(
                  (item) => item.status === "failed" || item.warnings.length > 0,
                ).length,
            }
          : current,
      );
      setNotice("검사가 끝났습니다. 경고 이미지는 삭제되지 않았으며 직접 결정할 수 있습니다.");
    } catch (error) {
      setNotice(String(error));
    } finally {
      setBusy(false);
    }
  }

  async function handleSaveTaskSpec() {
    if (!project) return;
    setBusy(true);
    setNotice("상황 설명과 결과 안전 규칙을 검증하고 있습니다.");
    try {
      await persistTaskConfiguration(true);
    } catch (error) {
      setNotice(String(error));
    } finally {
      setBusy(false);
    }
  }

  async function handleImportBackgrounds() {
    if (!project) return;
    const paths = await chooseBackgroundImages();
    if (paths.length === 0) return;
    setBusy(true);
    setNotice(`${paths.length}개 배경 이미지를 검사하고 있습니다.`);
    try {
      const imported = await importBackgroundImages(project.path, paths);
      setBackgrounds((current) => [...current, ...imported]);
      setProject((current) =>
        current
          ? {
              ...current,
              imageCount:
                current.imageCount +
                imported.filter((item) => item.status === "succeeded" && !item.duplicate).length,
              warningCount:
                current.warningCount +
                imported.filter(
                  (item) => item.status === "failed" || item.warnings.length > 0,
                ).length,
            }
          : current,
      );
      setNotice("배경 세트가 등록됐습니다. 실제 촬영 환경과 가까운 배경을 권장합니다.");
    } catch (error) {
      setNotice(String(error));
    } finally {
      setBusy(false);
    }
  }

  async function handleGenerate() {
    if (!project) return;
    setBusy(true);
    setNotice(
      `${generationCount}개 이미지를 한 장씩 생성합니다. 완료된 항목은 즉시 저장되고 메모리에서 해제됩니다.`,
    );
    try {
      const activeTask = await persistTaskConfiguration(false);
      const result = await generateSyntheticBatch(
        project.path,
        generationCount,
        generationSeed,
      );
      setGenerated((current) => [...current, ...result.items]);
      setProject((current) =>
        current
          ? {
              ...current,
              imageCount: current.imageCount + result.job.completedItems,
              warningCount: current.warningCount + result.job.failedItems,
            }
          : current,
      );
      setNotice(
        `합성 완료: 성공 ${result.job.completedItems}개, 실패 ${result.job.failedItems}개. TaskSpec r${activeTask?.revision ?? 0} · 시드 ${generationSeed}`,
      );
    } catch (error) {
      setNotice(String(error));
    } finally {
      setBusy(false);
    }
  }

  async function handleReview(
    assetIds: string[],
    reviewStatus: "approved" | "excluded",
  ) {
    if (!project || assetIds.length === 0) return;
    setBusy(true);
    setNotice(null);
    try {
      const update = await reviewImages(project.path, assetIds, reviewStatus);
      const selected = new Set(assetIds);
      setBackgrounds((current) =>
        current.map((item) =>
          selected.has(item.id) ? { ...item, reviewStatus: update.reviewStatus } : item,
        ),
      );
      setGenerated((current) =>
        current.map((item) =>
          item.id && selected.has(item.id)
            ? { ...item, reviewStatus: update.reviewStatus }
            : item,
        ),
      );
      setDataset(null);
      setModel(null);
      setInference(null);
      setNotice(
        `${update.updated}개 항목을 ${reviewStatus === "approved" ? "승인" : "제외"}했습니다. 기존 데이터셋 버전은 변경되지 않습니다.`,
      );
    } catch (error) {
      setNotice(String(error));
    } finally {
      setBusy(false);
    }
  }

  async function handleUpdateBoundingBox(assetId: string, boundingBox: BoundingBox) {
    if (!project) return;
    setBusy(true);
    setNotice(null);
    try {
      const saved = await updateBoundingBox(project.path, assetId, boundingBox);
      setGenerated((current) =>
        current.map((item) =>
          item.id === assetId
            ? { ...item, boundingBox: saved, reviewStatus: "needs_review" }
            : item,
        ),
      );
      setDataset(null);
      setModel(null);
      setInference(null);
      setNotice(
        "Box를 수정해 다시 승인 대기로 전환했습니다. 기존 데이터셋과 모델 파일은 보존되며 새 버전을 만들어야 수정 내용이 반영됩니다.",
      );
    } catch (error) {
      setNotice(String(error));
    } finally {
      setBusy(false);
    }
  }

  async function handleCreateDataset() {
    if (!project) return;
    setBusy(true);
    setNotice("승인 파일의 체크섬과 Box를 검사하고 불변 데이터셋을 만들고 있습니다.");
    try {
      const result = await createDatasetVersion(project.path, generationSeed);
      setDataset(result);
      setModel(null);
      setInference(null);
      setNotice(
        `데이터셋 v${String(result.version).padStart(4, "0")} 생성 완료: 학습 ${result.stats.trainItems}, 검증 ${result.stats.validationItems}, 테스트 ${result.stats.testItems}`,
      );
    } catch (error) {
      setNotice(String(error));
    } finally {
      setBusy(false);
    }
  }

  async function handleTrainModel() {
    if (!project || !dataset) return;
    setBusy(true);
    setNotice(
      "사전학습 탐지 모델을 로컬 장치에 맞춰 미세조정하고 있습니다. 중단 시 체크포인트에서 재개합니다.",
    );
    try {
      const result = await trainLocalModel(project.path, dataset.id, generationSeed);
      setModel(result);
      setInference(null);
      setNotice(
        `모델 학습 완료: Precision ${(result.metrics.precision * 100).toFixed(1)}%, Recall ${(result.metrics.recall * 100).toFixed(1)}%. 실제 촬영 세트로 반드시 검증하세요.`,
      );
    } catch (error) {
      setNotice(String(error));
    } finally {
      setBusy(false);
    }
  }

  async function handleInference() {
    if (!project || !model) return;
    const paths = await chooseInferenceImages();
    if (paths.length === 0) return;
    setBusy(true);
    setNotice(`${paths.length}개 실제 이미지를 한 장씩 판정하고 있습니다.`);
    try {
      await persistTaskConfiguration(false);
      const result = await runBatchInference(
        project.path,
        model.id,
        paths,
      );
      setInference(result);
      const routing = result.routing;
      setNotice(
        routing
          ? `분류 완료: 포함 ${routing.presentCount}, 미포함 ${routing.absentCount}, 검토 ${routing.reviewCount}, 실패 ${routing.failedCount}`
          : `일괄 추론 완료: 성공 ${result.job.completedItems}개, 실패 ${result.job.failedItems}개.`,
      );
    } catch (error) {
      setNotice(String(error));
    } finally {
      setBusy(false);
    }
  }

  async function handleExportModel() {
    if (!project || !model) return;
    const destination = await chooseModelExportPath(`VisionForge-${model.className}`);
    if (!destination) return;
    setBusy(true);
    setNotice("모델·파이프라인·지표·체크섬을 하나의 패키지로 만들고 다시 검증합니다.");
    try {
      const result = await exportModelPackage(project.path, model.id, destination);
      setNotice(
        `.vfmodel 내보내기 완료: ${result.packagePath} · SHA-256 ${result.packageChecksumSha256?.slice(0, 12)}…`,
      );
    } catch (error) {
      setNotice(String(error));
    } finally {
      setBusy(false);
    }
  }

  const summary = inspectionSummary(images);
  const usableTargets = images.filter((item) => item.status === "succeeded");
  const usableBackgrounds = backgrounds.filter((item) => item.status === "succeeded");
  const approvedGenerated = generated.filter(
    (item) => item.status === "succeeded" && item.reviewStatus === "approved",
  );

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-lockup">
          <span className="brand-mark" aria-hidden="true">VF</span>
          <div>
            <strong>VISIONFORGE</strong>
            <span>LOCAL VISION WORKBENCH</span>
          </div>
        </div>
        <div className="runtime-pill">
          <i className={status?.engineReady ? "ready" : "waiting"} />
          <span title={status?.hardware?.acceleratorName}>
            {status?.hardware?.profile === "APPLE_M1_16_BASELINE"
              ? "M1 16GB 기준"
              : status?.hardware?.accelerator === "cuda"
                ? "CUDA 가속"
                : status?.engineReady
                  ? "엔진 준비됨"
                  : "엔진 확인 중"}
          </span>
          <b>오프라인</b>
        </div>
      </header>

      <aside className="workflow-rail">
        <p className="eyebrow">WORKFLOW</p>
        <nav aria-label="VisionForge 작업 단계">
          {workflow.map(([number, label], index) => (
            <button className={index === 0 ? "workflow-step active" : "workflow-step"} key={number}>
              <span>{number}</span>
              {label}
            </button>
          ))}
        </nav>
        <div className="rail-note">
          <span>DATA POLICY</span>
          <strong>이미지는 이 컴퓨터를 벗어나지 않습니다.</strong>
        </div>
      </aside>

      <main className="workspace">
        <section className="hero-band">
          <div>
            <p className="eyebrow">IMAGE-ONLY / BATCH-FIRST</p>
            <h1>대상을 보여주면,<br />탐지 모델까지 이어집니다.</h1>
            <p className="hero-copy">
              여러 이미지를 한 작업으로 등록하고, 각 항목의 품질과 실패를 독립적으로 기록합니다.
            </p>
          </div>
          <div className="hero-orbit" aria-hidden="true">
            <span className="orbit-core">LOCAL</span>
            <span className="orbit-label one">검사</span>
            <span className="orbit-label two">합성</span>
            <span className="orbit-label three">학습</span>
          </div>
        </section>

        {notice && <div className="notice" role="status">{notice}</div>}

        {view === "setup" || !project ? (
          <>
            {project && (
              <section className="active-project-return">
                <div>
                  <span>현재 열려 있는 프로젝트</span>
                  <strong>{project.name}</strong>
                  <small>{project.path}</small>
                </div>
                <button type="button" onClick={handleResumeProject} disabled={busy}>
                  현재 프로젝트로 돌아가기
                  <span aria-hidden="true">→</span>
                </button>
              </section>
            )}
            <section className="setup-grid">
              <article className="panel create-panel">
                <div className="panel-heading">
                  <span>01</span>
                  <div>
                    <p className="eyebrow">NEW PROJECT</p>
                    <h2>첫 탐지 대상을 정의합니다.</h2>
                  </div>
                </div>
                <p className="setup-help">
                  빈 폴더를 미리 만들 필요가 없습니다. 상위 저장 폴더를 선택하면 그 안에
                  전용 프로젝트 폴더를 자동으로 만듭니다.
                </p>
                <label>
                  프로젝트 이름
                  <input value={projectName} onChange={(event) => setProjectName(event.target.value)} />
                </label>
                <label>
                  대상 클래스 이름
                  <input
                    value={className}
                    onChange={(event) => setClassName(event.target.value)}
                    placeholder="예: 빨간 부품 상자"
                  />
                </label>
                <button className="primary-button" onClick={handleCreateProject} disabled={busy}>
                  상위 저장 폴더를 선택하고 새 프로젝트 만들기
                  <span aria-hidden="true">↗</span>
                </button>
                <button className="open-project-button" onClick={handleOpenProject} disabled={busy}>
                  저장된 VisionForge 프로젝트 열기
                  <span aria-hidden="true">↙</span>
                </button>
                <small className="existing-project-help">
                  project.json과 project.sqlite가 들어 있는 프로젝트 폴더만 열 수 있습니다.
                </small>
              </article>

              <article className="panel principle-panel">
                <p className="eyebrow">HOW IT STORES</p>
                <h2>원본은 보존하고,<br />임시 변형은 쌓지 않습니다.</h2>
                <div className="storage-lines">
                  <div><span>영구</span><strong>원본 · 승인 합성본 · 모델</strong></div>
                  <div><span>제한</span><strong>썸네일 · 재생성 가능 캐시</strong></div>
                  <div><span>순간</span><strong>현재 학습 배치의 RAM 데이터</strong></div>
                </div>
                <button className="package-import" onClick={handleImportModelPackage} disabled={busy}>
                  <span>.VFMODEL</span>
                  <strong>공유 모델 가져와 바로 판정</strong>
                  <i aria-hidden="true">↘</i>
                </button>
              </article>
            </section>
          </>
        ) : (
          <>
            <section className="project-strip">
              <div className="project-identity">
                <p className="eyebrow">ACTIVE PROJECT</p>
                <h2>{project.name}</h2>
                <span className="project-path">{project.path}</span>
                <button
                  type="button"
                  className="change-project-button"
                  onClick={handleShowProjectSetup}
                  disabled={busy}
                >
                  <span aria-hidden="true">←</span>
                  프로젝트 선택 화면
                </button>
              </div>
              <div className="project-stat"><span>대상</span><strong>{project.className}</strong></div>
              <div className="project-stat"><span>등록</span><strong>{project.imageCount}</strong></div>
              <div className="project-stat"><span>검토</span><strong>{project.warningCount}</strong></div>
            </section>

            <section className="panel task-spec-panel">
              <div className="task-spec-copy">
                <p className="eyebrow">DYNAMIC TASK SPEC</p>
                <h2>실제 상황과 결과 규칙을 설명합니다.</h2>
                <p>
                  현재 v1은 사물 포함·미포함 작업을 지원합니다. 설명은 로컬 규칙 컴파일러가
                  크기·회전·조명·블러·노이즈·가림 생성 정책으로 변환합니다.
                </p>
                <label className="scenario-field">
                  실제로 전달될 사진의 상황
                  <textarea
                    value={scenarioDescription}
                    maxLength={4000}
                    onChange={(event) => setScenarioDescription(event.target.value)}
                    placeholder="예: 실내외에서 촬영되며 대상이 멀리 작게 보이거나 일부 가려질 수 있고 어두운 사진과 저화질 사진도 포함됨"
                  />
                  <small>{scenarioDescription.length} / 4000</small>
                </label>

                <div className="output-folder-grid">
                  <label>
                    포함 폴더
                    <input value={presentFolder} onChange={(event) => setPresentFolder(event.target.value)} />
                  </label>
                  <label>
                    미포함 폴더
                    <input value={absentFolder} onChange={(event) => setAbsentFolder(event.target.value)} />
                  </label>
                  <label>
                    검토 필요 폴더
                    <input value={reviewFolder} onChange={(event) => setReviewFolder(event.target.value)} />
                  </label>
                  <label>
                    처리 실패 폴더
                    <input value={failedFolder} onChange={(event) => setFailedFolder(event.target.value)} />
                  </label>
                </div>

                <div className="threshold-grid">
                  <label>
                    확정 포함 기준
                    <input
                      type="number"
                      min="0"
                      max="1"
                      step="0.01"
                      value={positiveThreshold}
                      onChange={(event) => setPositiveThreshold(Number(event.target.value))}
                    />
                  </label>
                  <label>
                    확정 미포함 기준
                    <input
                      type="number"
                      min="0"
                      max="1"
                      step="0.01"
                      value={negativeThreshold}
                      onChange={(event) => setNegativeThreshold(Number(event.target.value))}
                    />
                  </label>
                  <div>
                    <span>불확실 구간</span>
                    <strong>{negativeThreshold.toFixed(2)} 초과 · {positiveThreshold.toFixed(2)} 미만</strong>
                    <small>이 구간은 확정하지 않고 검토 필요 폴더로 보냅니다.</small>
                  </div>
                </div>

                <button className="primary-button task-save" onClick={handleSaveTaskSpec} disabled={busy}>
                  {taskFormMatchesSavedSpec() ? "작업 명세 저장됨" : "작업 명세 검증·저장"}
                  <span>r{taskSpec?.revision ?? 0}</span>
                </button>
              </div>

              <aside className="compiled-spec">
                <div className="compiled-head">
                  <span>COMPILED</span>
                  <strong>{taskSpec?.pipelineId ?? "single_class_object_detection"}</strong>
                </div>
                <div className="compiled-tags">
                  {(taskSpec?.compiledTags.length ?? 0) > 0
                    ? taskSpec?.compiledTags.map((tag) => <span key={tag}>{tag}</span>)
                    : <span>default_distribution</span>}
                </div>
                {taskSpec && (
                  <dl className="policy-readout">
                    <div><dt>크기</dt><dd>{taskSpec.generationPolicy.scaleMin.toFixed(2)}–{taskSpec.generationPolicy.scaleMax.toFixed(2)}</dd></div>
                    <div><dt>회전</dt><dd>{taskSpec.generationPolicy.rotationMin.toFixed(0)}°–{taskSpec.generationPolicy.rotationMax.toFixed(0)}°</dd></div>
                    <div><dt>밝기</dt><dd>{taskSpec.generationPolicy.brightnessMin.toFixed(2)}–{taskSpec.generationPolicy.brightnessMax.toFixed(2)}</dd></div>
                    <div><dt>블러</dt><dd>0–{taskSpec.generationPolicy.blurRadiusMax.toFixed(1)}</dd></div>
                    <div><dt>노이즈</dt><dd>0–{taskSpec.generationPolicy.noiseStddevMax.toFixed(2)}</dd></div>
                    <div><dt>가림</dt><dd>0–{(taskSpec.generationPolicy.occlusionMax * 100).toFixed(0)}%</dd></div>
                  </dl>
                )}
                {taskSpec?.warnings.map((warning) => <p key={warning}>{warning}</p>)}
                <small>자연어 원문과 컴파일 결과는 리비전별로 보존됩니다.</small>
              </aside>
            </section>

            <section className="panel import-panel">
              <div className="import-copy">
                <p className="eyebrow">TARGET SOURCES</p>
                <h2>대상 이미지 세트</h2>
                <p>다양한 각도와 거리의 사진을 함께 등록하세요. 파일은 자동 삭제되지 않습니다.</p>
              </div>
              <button className="drop-action" onClick={handleImportImages} disabled={busy}>
                <span className="plus">+</span>
                <strong>이미지 여러 장 선택</strong>
                <small>JPG · PNG · WEBP · BMP</small>
              </button>
            </section>

            <section className="synthesis-grid">
              <article className="panel background-panel">
                <p className="eyebrow">REAL-WORLD CONTEXT</p>
                <h2>실제 환경에 가까운 배경</h2>
                <p>
                  매장, 창고, 도로처럼 실제 추론 장소와 비슷한 사진을 사용하면 합성-실사 격차를 줄일 수 있습니다.
                </p>
                <button className="secondary-button" onClick={handleImportBackgrounds} disabled={busy}>
                  배경 이미지 세트 선택
                  <span>{usableBackgrounds.length}개 사용 가능</span>
                </button>
                {usableBackgrounds.length > 0 && (
                  <div className="review-toolbar">
                    <button
                      onClick={() => handleReview(usableBackgrounds.map((item) => item.id), "approved")}
                      disabled={busy}
                    >
                      사용 가능한 배경 전체 승인
                    </button>
                    <span>
                      승인 {usableBackgrounds.filter((item) => item.reviewStatus === "approved").length}
                    </span>
                  </div>
                )}
                {backgrounds.length > 0 && (
                  <div className="compact-files">
                    {backgrounds.slice(-4).map((background, index) => (
                      <span key={`${background.id}-${index}`}>{background.originalName}</span>
                    ))}
                  </div>
                )}
              </article>

              <article className="panel generation-panel">
                <p className="eyebrow">DETERMINISTIC SYNTHESIS</p>
                <h2>검토할 기본 합성본 생성</h2>
                <div className="generation-fields">
                  <label>
                    생성 수
                    <input
                      type="number"
                      min="1"
                      max="10000"
                      value={generationCount}
                      onChange={(event) => setGenerationCount(Number(event.target.value))}
                    />
                  </label>
                  <label>
                    랜덤 시드
                    <input
                      type="number"
                      value={generationSeed}
                      onChange={(event) => setGenerationSeed(Number(event.target.value))}
                    />
                  </label>
                </div>
                <div className="memory-policy">
                  <span>RAM</span>
                  <strong>현재 이미지 1장만 처리</strong>
                  <span>DISK</span>
                  <strong>검토할 합성본과 레시피만 보존</strong>
                </div>
                <button
                  className="primary-button"
                  onClick={handleGenerate}
                  disabled={busy || usableTargets.length === 0 || usableBackgrounds.length === 0}
                >
                  합성 작업 시작
                  <span aria-hidden="true">→</span>
                </button>
              </article>
            </section>

            {images.length > 0 && (
              <section className="inspection-section">
                <div className="inspection-head">
                  <div>
                    <p className="eyebrow">LOCAL INSPECTION</p>
                    <h2>이미지별 검사 결과</h2>
                  </div>
                  <div className="result-legend">
                    <span className="good">사용 가능 {summary.ready}</span>
                    <span className="review">검토 {summary.review}</span>
                    <span className="failed">실패 {summary.failed}</span>
                  </div>
                </div>
                <div className="image-grid">
                  {images.map((image, index) => (
                    <article className={`image-card ${image.status}`} key={`${image.id}-${index}`}>
                      <div className="image-sequence">{String(index + 1).padStart(2, "0")}</div>
                      <div className="file-glyph" aria-hidden="true">
                        <span>{image.status === "failed" ? "!" : "IMG"}</span>
                      </div>
                      <div className="image-meta">
                        <h3>{image.originalName}</h3>
                        <p>
                          {image.width && image.height ? `${image.width} × ${image.height}` : "디코딩 실패"}
                          <span>{compactChecksum(image.checksumSha256)}</span>
                        </p>
                        <div className="warning-stack">
                          {image.duplicate && <span>완전 중복</span>}
                          {image.warnings.slice(0, 2).map((warning) => (
                            <span key={warning.code}>{warning.message}</span>
                          ))}
                          {image.errorMessage && <span>{image.errorMessage}</span>}
                          {image.warnings.length === 0 && !image.duplicate && image.status === "succeeded" && (
                            <span className="clear">기본 검사 통과</span>
                          )}
                        </div>
                      </div>
                    </article>
                  ))}
                </div>
              </section>
            )}

            {generated.length > 0 && (
              <section className="inspection-section generated-section">
                <div className="inspection-head">
                  <div>
                    <p className="eyebrow">SYNTHETIC REVIEW SET</p>
                    <h2>생성 결과와 자동 Box</h2>
                  </div>
                  <div className="review-actions">
                    <span className="generated-count">{generated.length}개 생성됨</span>
                    <button
                      onClick={() =>
                        handleReview(
                          generated
                            .filter((item) => item.status === "succeeded" && item.id)
                            .map((item) => item.id as string),
                          "approved",
                        )
                      }
                      disabled={busy}
                    >
                      성공 항목 전체 승인
                    </button>
                  </div>
                </div>
                <div className="image-grid">
                  {generated.map((image, index) => (
                    <article className={`image-card ${image.status}`} key={`${image.id}-${index}`}>
                      <div className="image-sequence">S{String(index + 1).padStart(3, "0")}</div>
                      <div className="file-glyph generated-glyph" aria-hidden="true">
                        <span>BOX</span>
                      </div>
                      <div className="image-meta">
                        <h3>{image.outputPath.split(/[\\/]/).at(-1)}</h3>
                        <p>
                          시드 {image.seed}
                          <span>
                            {image.boundingBox
                              ? `Box ${image.boundingBox.xMin},${image.boundingBox.yMin} → ${image.boundingBox.xMax},${image.boundingBox.yMax}`
                              : "Box 생성 실패"}
                          </span>
                        </p>
                        <div className="warning-stack">
                          {image.status === "succeeded" ? (
                            <span className={image.reviewStatus === "approved" ? "clear" : ""}>
                              {image.reviewStatus === "approved" ? "학습 승인됨" : "검토 대기"}
                            </span>
                          ) : (
                            <span>{image.errorMessage ?? "생성 실패"}</span>
                          )}
                        </div>
                        {image.status === "succeeded" && image.id && (
                          <BoundingBoxEditor
                            image={image}
                            busy={busy}
                            onSave={(box) => handleUpdateBoundingBox(image.id as string, box)}
                          />
                        )}
                        {image.status === "succeeded" && image.id && (
                          <div className="card-actions">
                            <button
                              onClick={() => handleReview([image.id as string], "approved")}
                              disabled={busy || image.reviewStatus === "approved"}
                            >
                              승인
                            </button>
                            <button
                              onClick={() => handleReview([image.id as string], "excluded")}
                              disabled={busy || image.reviewStatus === "excluded"}
                            >
                              제외
                            </button>
                          </div>
                        )}
                      </div>
                    </article>
                  ))}
                </div>
              </section>
            )}

            {generated.length > 0 && (
              <section className="panel dataset-panel">
                <div>
                  <p className="eyebrow">IMMUTABLE DATASET</p>
                  <h2>승인 결과를 데이터셋 버전으로 고정</h2>
                  <p>
                    같은 원본에서 파생된 합성본은 같은 분할에 배치해 데이터 누수를 막습니다.
                    파일 체크섬과 Box 범위를 다시 검사하며 기존 버전은 절대 덮어쓰지 않습니다.
                  </p>
                </div>
                <div className="dataset-action">
                  <div>
                    <span>승인 양성</span>
                    <strong>{approvedGenerated.length}</strong>
                    <span>승인 부정</span>
                    <strong>
                      {usableBackgrounds.filter((item) => item.reviewStatus === "approved").length}
                    </strong>
                  </div>
                  <button
                    className="primary-button"
                    onClick={handleCreateDataset}
                    disabled={busy || approvedGenerated.length === 0}
                  >
                    새 데이터셋 버전 생성
                    <span aria-hidden="true">→</span>
                  </button>
                </div>
                {dataset && (
                  <div className="dataset-result">
                    <strong>v{String(dataset.version).padStart(4, "0")}</strong>
                    <span>학습 {dataset.stats.trainItems}</span>
                    <span>검증 {dataset.stats.validationItems}</span>
                    <span>테스트 {dataset.stats.testItems}</span>
                    <small>{dataset.manifestPath}</small>
                    {dataset.warnings.map((warning) => <p key={warning}>{warning}</p>)}
                  </div>
                )}
              </section>
            )}

            {(dataset || model) && (
              <section className="model-stage">
                <article className="panel model-copy">
                  <p className="eyebrow">
                    {dataset ? "TRANSFER LEARNING / AUTO ACCELERATOR" : "VERIFIED MODEL PACKAGE"}
                  </p>
                  <h2>
                    {dataset ? "사전학습 비전 모델 로컬 미세조정" : "재학습 없이 준비된 공유 모델"}
                  </h2>
                  <p>
                    {dataset
                      ? "M1 Pro 16GB에서는 batch 1과 누적 기울기를 사용합니다. 변형 텐서는 메모리에서 해제하고 30분 이내 간격으로 재개 지점을 저장합니다."
                      : "패키지 내부 경로, 압축 해제 한도, 파일별 SHA-256, 엔진 호환성을 통과한 모델입니다."}
                  </p>
                  <div className="baseline-warning">
                    <strong>실제사진 검증</strong>
                    <span>
                      학습 완료와 배포 가능은 다릅니다. 실제 촬영 고정 평가 세트의 품질 게이트를 통과해야 합니다.
                    </span>
                  </div>
                  {dataset ? (
                    <button className="primary-button" onClick={handleTrainModel} disabled={busy}>
                      데이터셋 v{String(dataset.version).padStart(4, "0")} 학습
                      <span aria-hidden="true">→</span>
                    </button>
                  ) : (
                    <div className="imported-model-ready">
                      <span>CLASS</span>
                      <strong>{model?.className}</strong>
                    </div>
                  )}
                </article>

                <article className="panel metric-panel">
                  <p className="eyebrow">FIXED EVALUATION</p>
                  {model ? (
                    <>
                      <div className="model-id">
                        <span>
                          {model.deploymentStatus === "qualified"
                            ? "DEPLOY READY"
                            : "VALIDATION ONLY"}
                        </span>
                        <strong>{model.id.slice(0, 8)}</strong>
                      </div>
                      <div className="metric-grid">
                        <div><span>Precision</span><strong>{(model.metrics.precision * 100).toFixed(1)}%</strong></div>
                        <div><span>Recall</span><strong>{(model.metrics.recall * 100).toFixed(1)}%</strong></div>
                        <div><span>F1</span><strong>{(model.metrics.f1 * 100).toFixed(1)}%</strong></div>
                        <div><span>평균 IoU</span><strong>{(model.metrics.meanIou * 100).toFixed(1)}%</strong></div>
                      </div>
                      <small>{model.metrics.evaluationSplit === "train_fallback" ? "학습 세트 기본 확인" : "고정 검증 세트"}</small>
                      {model.warnings.map((warning) => <p key={warning.code}>{warning.message}</p>)}
                      <button className="model-export" onClick={handleExportModel} disabled={busy}>
                        .vfmodel 내보내기
                        <span>무결성 재검증 포함</span>
                      </button>
                    </>
                  ) : (
                    <div className="empty-metric">
                      <strong>아직 모델이 없습니다.</strong>
                      <span>학습이 끝나면 검증 지표와 모델 체크섬을 이곳에 고정합니다.</span>
                    </div>
                  )}
                </article>
              </section>
            )}

            {model && (
              <section className="inference-section">
                <div className="inference-heading">
                  <div>
                    <p className="eyebrow">REAL IMAGE CHECK</p>
                    <h2>실제 사진 일괄 판정</h2>
                    <p>합성 이미지 점수와 별개로, 실제 촬영 사진에서 괴리를 확인합니다.</p>
                  </div>
                  <div className="inference-control">
                    <div className="decision-bands">
                      <span>포함 ≥ {positiveThreshold.toFixed(2)}</span>
                      <span>미포함 ≤ {negativeThreshold.toFixed(2)}</span>
                      <strong>사이 값은 검토 필요</strong>
                    </div>
                    <button className="primary-button" onClick={handleInference} disabled={busy}>
                      실제 이미지 여러 장 선택
                      <span aria-hidden="true">↗</span>
                    </button>
                  </div>
                </div>

                {inference && (
                  <>
                    {inference.routing && (
                      <div className="routing-summary">
                        <div><span>포함</span><strong>{inference.routing.presentCount}</strong></div>
                        <div><span>미포함</span><strong>{inference.routing.absentCount}</strong></div>
                        <div><span>검토</span><strong>{inference.routing.reviewCount}</strong></div>
                        <div><span>실패</span><strong>{inference.routing.failedCount}</strong></div>
                        <p>{inference.routing.rootPath}</p>
                      </div>
                    )}
                    <div className="inference-grid">
                      {inference.items.map((item, index) => (
                        <article className={`inference-card ${item.status} ${item.decision}`} key={item.id}>
                          <span className="inference-index">R{String(index + 1).padStart(3, "0")}</span>
                          <h3>{item.inputPath}</h3>
                          <span className={`decision-badge ${item.decision}`}>
                            {decisionLabel(item.decision)}
                          </span>
                          {item.status === "succeeded" ? (
                            <>
                              <strong className="detection-count">
                                {item.detections.length === 0
                                  ? "확정 Box 없음"
                                  : `후보 ${item.detections.length}개`}
                              </strong>
                              <div className="detection-list">
                                {item.detections.slice(0, 5).map((detection, detectionIndex) => (
                                  <span key={`${item.id}-${detectionIndex}`}>
                                    {(detection.confidence * 100).toFixed(1)}% · [{detection.boundingBox.xMin}, {detection.boundingBox.yMin}, {detection.boundingBox.xMax}, {detection.boundingBox.yMax}]
                                  </span>
                                ))}
                              </div>
                              <small>
                                최고 후보 {item.maxConfidence === null ? "없음" : `${(item.maxConfidence * 100).toFixed(1)}%`} · {item.elapsedMs?.toFixed(0)} ms
                              </small>
                            </>
                          ) : (
                            <p>{item.errorMessage ?? "이미지 처리 실패"}</p>
                          )}
                          {item.routedPath && <small className="route-path">{item.routedPath}</small>}
                          {item.routingError && <p>{item.routingError}</p>}
                        </article>
                      ))}
                    </div>
                  </>
                )}
              </section>
            )}
          </>
        )}
      </main>
    </div>
  );
}

export default App;
