import { JobStatus, NullableJobStatus } from "../types/prediction";

export type StatusTone = "warning" | "info" | "success" | "danger";

export const STATUS_COPY: Record<JobStatus, { label: string; tone: StatusTone }> = {
  queued: { label: "Queued", tone: "warning" },
  processing: { label: "Processing", tone: "info" },
  done: { label: "Completed", tone: "success" },
  error: { label: "Failed", tone: "danger" },
  saved: { label: "Saved", tone: "success" },
};

const STATUS_LOOKUP: Record<string, JobStatus> = {
  queued: "queued",
  processing: "processing",
  running: "processing",
  done: "done",
  error: "error",
  saved: "saved",
};

export function normalizeStatus(value: unknown): NullableJobStatus {
  if (typeof value !== "string") {
    return null;
  }

  const key = value.toLowerCase();
  return STATUS_LOOKUP[key] ?? null;
}
