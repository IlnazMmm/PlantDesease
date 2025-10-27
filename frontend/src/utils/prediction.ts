export const CONFIDENCE_WARNING_THRESHOLD = 0.6;

export type ConfidenceTone = "high" | "medium" | "low";

export function getConfidenceTone(confidence: number): ConfidenceTone {
  if (confidence >= 0.8) {
    return "high";
  }

  if (confidence >= CONFIDENCE_WARNING_THRESHOLD) {
    return "medium";
  }

  return "low";
}

export function formatConfidence(confidence?: number | null): string {
  if (confidence == null) {
    return "—";
  }

  return `${(confidence * 100).toFixed(1)}%`;
}

export function formatDateTime(iso?: string | null): string {
  if (!iso) {
    return "";
  }

  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) {
    return "";
  }

  return date.toLocaleString("ru-RU", {
    dateStyle: "short",
    timeStyle: "short",
  });
}
