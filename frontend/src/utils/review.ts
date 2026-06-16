import { ReviewStatus } from "../types/prediction";

export const REVIEW_STATUS_LABELS: Record<ReviewStatus, string> = {
  not_required: "Подтверждение не требуется",
  pending: "Ожидает подтверждения",
  confirmed: "Подтверждено",
  corrected: "Скорректировано экспертом",
};

export function getReviewStatusLabel(status?: ReviewStatus | null): string {
  return REVIEW_STATUS_LABELS[status ?? "not_required"];
}
