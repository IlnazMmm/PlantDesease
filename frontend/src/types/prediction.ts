export type JobStatus = "queued" | "processing" | "done" | "error" | "saved";
export type NullableJobStatus = JobStatus | null;

export type ReviewStatus = "not_required" | "pending" | "confirmed" | "corrected";

export interface PredictionResult {
  plant: string;
  disease: string;
  confidence: number;
  gradcam_url?: string;
  description?: string;
  treatment?: string;
  prevention?: string;
  pathogen?: string;
  label?: string;
  job_id?: string;
  created_at?: string;
  review_required?: boolean;
  review_status?: ReviewStatus;
  expert_label?: string | null;
  expert_comment?: string | null;
  reviewed_at?: string | null;
  review_warning?: string | null;
}

export interface HistoryItem {
  job_id: string;
  plant?: string;
  disease?: string;
  confidence?: number;
  gradcam_url?: string | null;
  created_at?: string;
  label?: string | null;
  review_required?: boolean;
  review_status?: ReviewStatus;
  expert_label?: string | null;
  expert_comment?: string | null;
  reviewed_at?: string | null;
}


export interface ReviewPayload {
  confirmed: boolean;
  expert_label?: string | null;
  expert_comment?: string | null;
}

export interface ReviewResponse extends ReviewPayload {
  status: string;
  job_id: string;
  review_required: boolean;
  review_status: ReviewStatus;
  reviewed_at?: string | null;
}
