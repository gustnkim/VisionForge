export type WarningItem = {
  code: string;
  message: string;
  value: string | number | null;
};

export type ProjectSummary = {
  id: string;
  name: string;
  classId: string;
  className: string;
  path: string;
  createdAt: string;
  imageCount: number;
  warningCount: number;
};

export type OutputPolicy = {
  presentFolder: string;
  absentFolder: string;
  reviewFolder: string;
  failedFolder: string;
  positiveThreshold: number;
  negativeThreshold: number;
  copyMode: "copy_original";
};

export type GenerationPolicySpec = {
  scaleMin: number;
  scaleMax: number;
  rotationMin: number;
  rotationMax: number;
  brightnessMin: number;
  brightnessMax: number;
  contrastMin: number;
  contrastMax: number;
  blurRadiusMax: number;
  noiseStddevMax: number;
  occlusionMax: number;
};

export type TaskSpec = {
  schemaVersion: number;
  id: string;
  revision: number;
  taskType: "object_presence";
  pipelineId: string;
  classId: string;
  className: string;
  scenarioDescription: string;
  compiledTags: string[];
  generationPolicy: GenerationPolicySpec;
  outputPolicy: OutputPolicy;
  compiler: string;
  warnings: string[];
  createdAt: string;
};

export type TaskSpecInput = {
  taskType: "object_presence";
  scenarioDescription: string;
  outputPolicy: OutputPolicy;
};

export type ImportedImage = {
  id: string;
  role: "target_original" | "background" | string;
  originalPath: string;
  originalName: string;
  internalPath: string | null;
  status: "succeeded" | "failed";
  width: number | null;
  height: number | null;
  checksumSha256: string | null;
  warnings: WarningItem[];
  duplicate: boolean;
  reviewStatus: "approved" | "excluded" | "unreviewed" | "needs_review";
  errorCode: string | null;
  errorMessage: string | null;
};

export type BoundingBox = {
  xMin: number;
  yMin: number;
  xMax: number;
  yMax: number;
};

export type GeneratedImage = {
  id: string | null;
  status: "succeeded" | "failed";
  outputPath: string;
  seed: number;
  boundingBox: BoundingBox | null;
  warnings: WarningItem[];
  reviewStatus: "approved" | "excluded" | "unreviewed" | "needs_review";
  errorCode: string | null;
  errorMessage: string | null;
};

export type JobSummary = {
  id: string;
  jobType: string;
  status: "running" | "succeeded" | "partial_failed" | "failed";
  totalItems: number;
  completedItems: number;
  failedItems: number;
  createdAt: string;
  updatedAt: string;
};

export type GenerationBatchResult = {
  job: JobSummary;
  items: GeneratedImage[];
};

export type ReviewUpdate = {
  requested: number;
  updated: number;
  reviewStatus: "approved" | "excluded" | "unreviewed" | "needs_review";
};

export type DatasetStats = {
  totalItems: number;
  positiveItems: number;
  negativeItems: number;
  trainItems: number;
  validationItems: number;
  testItems: number;
};

export type DatasetVersionSummary = {
  id: string;
  version: number;
  manifestPath: string;
  checksumSha256: string;
  createdAt: string;
  taskSpecId: string;
  taskSpecRevision: number;
  stats: DatasetStats;
  warnings: string[];
};

export type TrainingMetrics = {
  evaluationSplit: string;
  positiveImages: number;
  negativeImages: number;
  truePositives: number;
  falsePositives: number;
  falseNegatives: number;
  precision: number;
  recall: number;
  f1: number;
  meanIou: number;
};

export type ModelVersionSummary = {
  id: string;
  datasetId: string;
  status: "ready";
  deploymentStatus: "experimental" | "candidate" | "qualified";
  engineName: string;
  classId: string;
  className: string;
  origin: "trained" | "imported";
  modelPath: string;
  checksumSha256: string;
  metrics: TrainingMetrics;
  warnings: WarningItem[];
  createdAt: string;
};

export type ModelPackageResult = {
  status: "succeeded" | "failed";
  packagePath: string;
  packageId: string | null;
  packageChecksumSha256: string | null;
  classId: string | null;
  className: string | null;
  engineName: string | null;
  deploymentStatus: "experimental" | "candidate" | "qualified" | null;
  modelPath: string | null;
  metricsPath: string | null;
  taskSpecPath: string | null;
  warnings: WarningItem[];
  errorCode: string | null;
  errorMessage: string | null;
};

export type ImportedModelProject = {
  project: ProjectSummary;
  model: ModelVersionSummary;
};

export type ProjectWorkspace = {
  project: ProjectSummary;
  targets: ImportedImage[];
  backgrounds: ImportedImage[];
  generated: GeneratedImage[];
  dataset: DatasetVersionSummary | null;
  model: ModelVersionSummary | null;
  taskSpec: TaskSpec;
  recoveredJobs: number;
};

export type Detection = {
  classId: string;
  className: string;
  confidence: number;
  boundingBox: BoundingBox;
};

export type InferenceItemRecord = {
  id: string;
  status: "succeeded" | "failed";
  inputPath: string;
  outputPath: string | null;
  detections: Detection[];
  maxConfidence: number | null;
  decision: "present" | "absent" | "review" | "failed" | "unrouted";
  routedPath: string | null;
  routingError: string | null;
  elapsedMs: number | null;
  errorCode: string | null;
  errorMessage: string | null;
};

export type RoutingSummary = {
  rootPath: string;
  manifestPath: string;
  presentCount: number;
  absentCount: number;
  reviewCount: number;
  failedCount: number;
};

export type InferenceBatchResult = {
  runId: string;
  modelId: string;
  job: JobSummary;
  items: InferenceItemRecord[];
  routing: RoutingSummary | null;
};

export type SystemStatus = {
  offline: boolean;
  engineReady: boolean;
  enginePath: string;
  platform: string;
  hardware: HardwareProfile | null;
};

export type HardwareProfile = {
  profile: "APPLE_M1_16_BASELINE" | "APPLE_HIGH_MEMORY" | "CUDA_ACCELERATED" | "CPU_FALLBACK";
  platform: string;
  architecture: string;
  cpuCount: number;
  totalMemoryBytes: number | null;
  accelerator: "mps" | "cuda" | "cpu";
  acceleratorName: string;
  acceleratorMemoryBytes: number | null;
  executionProviders: string[];
  freeDiskBytes: number | null;
  torchVersion: string | null;
  torchvisionVersion: string | null;
};
