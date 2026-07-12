import { existsSync } from "node:fs";
import { readdir, stat } from "node:fs/promises";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

import {
  sidecarDestination,
  sidecarRuntimeDirectory,
} from "./build-engine-sidecar.mjs";

const scriptDirectory = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(scriptDirectory, "..");
const desktop = path.join(root, "apps", "desktop");

function run(executable, args, cwd = root) {
  const result = spawnSync(executable, args, {
    cwd,
    encoding: "utf8",
    stdio: "inherit",
    env: process.env,
  });
  if (result.error) throw result.error;
  if (result.status !== 0) {
    throw new Error(`${path.basename(executable)} failed with exit code ${result.status}.`);
  }
}

async function latestModifiedAt(target) {
  const targetStat = await stat(target);
  if (!targetStat.isDirectory()) return targetStat.mtimeMs;
  const entries = await readdir(target, { withFileTypes: true });
  const values = await Promise.all(
    entries.map((entry) => latestModifiedAt(path.join(target, entry.name))),
  );
  return Math.max(targetStat.mtimeMs, ...values);
}

function packageExecutable(relativePath) {
  const candidates = [
    path.join(root, "node_modules", relativePath),
    path.join(desktop, "node_modules", relativePath),
  ];
  const executable = candidates.find((candidate) => existsSync(candidate));
  if (!executable) {
    throw new Error(`Frontend dependency is missing: ${relativePath}`);
  }
  return executable;
}

async function prepare() {
  const sidecar = sidecarDestination();
  const sources = [
    path.join(engineRoot(), "src"),
    path.join(engineRoot(), "resources"),
    path.join(engineRoot(), "sidecar.py"),
    path.join(engineRoot(), "pyproject.toml"),
    path.join(engineRoot(), "uv.lock"),
    path.join(scriptDirectory, "build-engine-sidecar.mjs"),
  ];
  const latestSource = Math.max(
    ...(await Promise.all(sources.map((source) => latestModifiedAt(source)))),
  );
  const bundledWeight = path.join(
    sidecarRuntimeDirectory(),
    "_internal",
    "resources",
    "weights",
    "fasterrcnn_mobilenet_v3_large_fpn-fb6a3cc7.pth",
  );
  const sidecarIsCurrent =
    existsSync(sidecar) &&
    existsSync(bundledWeight) &&
    (await stat(sidecar)).mtimeMs >= latestSource;
  if (!sidecarIsCurrent) {
    run(process.execPath, [path.join(scriptDirectory, "build-engine-sidecar.mjs")]);
  }

  run(
    process.execPath,
    [packageExecutable(path.join("typescript", "bin", "tsc")), "-b", "--pretty", "false"],
    desktop,
  );
  run(
    process.execPath,
    [packageExecutable(path.join("vite", "bin", "vite.js")), "build"],
    desktop,
  );
}

function engineRoot() {
  return path.join(root, "engine");
}

prepare().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
