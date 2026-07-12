import { describe, expect, it } from "vitest";

import { compactChecksum, inspectionSummary } from "./domain";
import type { ImportedImage } from "./types";

function image(overrides: Partial<ImportedImage> = {}): ImportedImage {
  return {
    id: "image-1",
    role: "target_original",
    originalPath: "C:/target.png",
    originalName: "target.png",
    internalPath: "assets/originals/hash.png",
    status: "succeeded",
    width: 320,
    height: 240,
    checksumSha256: "1234567890abcdef",
    warnings: [],
    duplicate: false,
    reviewStatus: "unreviewed",
    errorCode: null,
    errorMessage: null,
    ...overrides,
  };
}

describe("inspectionSummary", () => {
  it("keeps ready, review, and failure counts independent", () => {
    const result = inspectionSummary([
      image(),
      image({ warnings: [{ code: "blur", message: "흐림", value: 10 }] }),
      image({ status: "failed", errorCode: "decode_failed" }),
    ]);
    expect(result).toEqual({ ready: 1, review: 1, failed: 1 });
  });
});

describe("compactChecksum", () => {
  it("does not expose the full checksum in the table", () => {
    expect(compactChecksum("1234567890abcdef")).toBe("12345678...abcdef");
  });
});
