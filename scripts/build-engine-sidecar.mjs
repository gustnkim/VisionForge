import { randomUUID } from "node:crypto";
import { existsSync } from "node:fs";
import { chmod, cp, mkdir, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const scriptDirectory = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(scriptDirectory, "..");
const engine = path.join(root, "engine");

function run(executable, args, options = {}) {
  const result = spawnSync(executable, args, {
    cwd: options.cwd ?? root,
    encoding: "utf8",
    stdio: options.capture ? "pipe" : "inherit",
    env: process.env,
  });
  if (result.error) {
    throw result.error;
  }
  if (result.status !== 0) {
    const detail = options.capture ? `\n${result.stderr || result.stdout}` : "";
    throw new Error(`${path.basename(executable)} failed with exit code ${result.status}.${detail}`);
  }
  return options.capture ? result.stdout.trim() : "";
}

function firstExisting(candidates) {
  return candidates.find((candidate) => candidate && existsSync(candidate));
}

function pythonExecutable() {
  const configured = process.env.VISIONFORGE_BUILD_PYTHON;
  const executable = firstExisting([
    configured,
    process.platform === "win32"
      ? path.join(engine, ".venv", "Scripts", "python.exe")
      : path.join(engine, ".venv", "bin", "python"),
  ]);
  if (!executable) {
    throw new Error(
      "The engine virtual environment is missing. Run `uv sync --project engine --all-groups` first.",
    );
  }
  return executable;
}

export function sidecarRuntimeDirectory() {
  return path.join(
    root,
    "apps",
    "desktop",
    "src-tauri",
    "binaries",
    "visionforge-engine-runtime",
  );
}

export function sidecarDestination() {
  const extension = process.platform === "win32" ? ".exe" : "";
  return path.join(sidecarRuntimeDirectory(), `visionforge-engine${extension}`);
}

async function build() {
  const python = pythonExecutable();
  const buildRoot = path.join(root, "target", "pyinstaller-work", randomUUID());
  const specPath = path.join(buildRoot, "spec");
  const workPath = path.join(buildRoot, "work");
  const distPath = path.join(buildRoot, "dist");
  await mkdir(specPath, { recursive: true });
  await mkdir(workPath, { recursive: true });
  await mkdir(distPath, { recursive: true });

  const resourceArgument = `${path.join(engine, "resources")}${path.delimiter}resources`;
  try {
    run(
      python,
      [
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onedir",
        "--console",
        "--specpath",
        specPath,
        "--workpath",
        workPath,
        "--distpath",
        distPath,
        "--name",
        "visionforge-engine",
        "--paths",
        path.join(engine, "src"),
        "--add-data",
        resourceArgument,
        "--collect-data",
        "torch",
        "--collect-binaries",
        "torch",
        "--collect-data",
        "torchvision",
        "--collect-binaries",
        "torchvision",
        "--copy-metadata",
        "torch",
        "--copy-metadata",
        "torchvision",
        path.join(engine, "sidecar.py"),
      ],
      { cwd: engine },
    );

    const extension = process.platform === "win32" ? ".exe" : "";
    const sourceDirectory = path.join(distPath, "visionforge-engine");
    const source = path.join(sourceDirectory, `visionforge-engine${extension}`);
    if (!existsSync(source)) {
      throw new Error(`PyInstaller output is missing: ${source}`);
    }
    const bundledWeight = path.join(
      sourceDirectory,
      "_internal",
      "resources",
      "weights",
      "fasterrcnn_mobilenet_v3_large_fpn-fb6a3cc7.pth",
    );
    if (!existsSync(bundledWeight)) {
      throw new Error(`Bundled pretrained weight is missing: ${bundledWeight}`);
    }
    const runtimeDirectory = sidecarRuntimeDirectory();
    await rm(runtimeDirectory, { recursive: true, force: true });
    await cp(sourceDirectory, runtimeDirectory, { recursive: true, force: true });
    await writeFile(path.join(runtimeDirectory, ".gitkeep"), "", "utf8");
    const destination = sidecarDestination();
    if (process.platform !== "win32") {
      await chmod(destination, 0o755);
    }
    console.log(`VisionForge engine sidecar: ${destination}`);
  } finally {
    await rm(buildRoot, { recursive: true, force: true });
  }
}

const invokedDirectly = process.argv[1]
  ? path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)
  : false;
if (invokedDirectly) {
  build().catch((error) => {
    console.error(error instanceof Error ? error.message : error);
    process.exitCode = 1;
  });
}
