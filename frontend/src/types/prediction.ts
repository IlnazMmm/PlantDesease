export type JobStatus = "queued" | "processing" | "done" | "error" | "saved";
export type NullableJobStatus = JobStatus | null;

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
}

export interface HistoryItem {
  job_id: string;
  plant?: string;
  disease?: string;
  confidence?: number;
  gradcam_url?: string | null;
  created_at?: string;
  label?: string | null;
}
