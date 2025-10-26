import React from "react";

import { NullableJobStatus } from "../types/prediction";
import { STATUS_COPY } from "../utils/status";

interface StatusBannerProps {
  status: NullableJobStatus;
  jobId: string | null;
  onCopy: () => void;
  copyFeedback: string | null;
}

export function StatusBanner({ status, jobId, onCopy, copyFeedback }: StatusBannerProps) {
  if (!status) {
    return null;
  }

  const config = STATUS_COPY[status];

  return (
    <div className="status">
      <span className={`status__badge status__badge--${config?.tone ?? "info"}`}>
        {config?.label ?? status}
      </span>
      {jobId && (
        <div className="status__job">
          <span className="status__job-label">ID задачи:</span>
          <span
            className="status__job-id"
            role="button"
            tabIndex={0}
            title="Скопировать ID задачи"
            aria-label="Скопировать ID задачи"
            onClick={onCopy}
            onKeyDown={(event) => {
              if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                onCopy();
              }
            }}
          >
            {jobId}
          </span>
          {copyFeedback && <span className="status__copy-feedback">{copyFeedback}</span>}
        </div>
      )}
    </div>
  );
}
