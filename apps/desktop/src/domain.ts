import type { ImportedImage } from "./types";

export function inspectionSummary(images: ImportedImage[]) {
  return images.reduce(
    (summary, image) => {
      if (image.status === "failed") summary.failed += 1;
      else if (image.warnings.length > 0 || image.duplicate) summary.review += 1;
      else summary.ready += 1;
      return summary;
    },
    { ready: 0, review: 0, failed: 0 },
  );
}

export function compactChecksum(checksum: string | null) {
  return checksum ? `${checksum.slice(0, 8)}...${checksum.slice(-6)}` : "체크섬 없음";
}

