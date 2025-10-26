import React, { useCallback, useEffect, useRef, useState } from "react";
import axios from "axios";
import api, { API_BASE_URL } from "../services/api";

type PredictionResult = {
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
};

type HistoryItem = {
  job_id: string;
  plant?: string;
  disease?: string;
  confidence?: number;
  gradcam_url?: string | null;
  created_at?: string;
  label?: string | null;
};

type JobStatus = "queued" | "processing" | "done" | "error" | "saved" | null;

const CONFIDENCE_WARNING_THRESHOLD = 0.6;

type ConfidenceTone = "high" | "medium" | "low";

const STATUS_COPY: Record<Exclude<JobStatus, null>, { label: string; tone: "warning" | "info" | "success" | "danger" }> = {
  queued: { label: "Queued", tone: "warning" },
  processing: { label: "Processing", tone: "info" },
  done: { label: "Completed", tone: "success" },
  error: { label: "Failed", tone: "danger" },
  saved: { label: "Saved", tone: "success" },
};

const normalizeStatus = (value: unknown): JobStatus => {
  if (typeof value !== "string") {
    return null;
  }

  const lookup: Record<string, JobStatus> = {
    queued: "queued",
    processing: "processing",
    running: "processing",
    done: "done",
    error: "error",
    saved: "saved",
  };

  return lookup[value] ?? null;
};

export default function Home(): JSX.Element {
  const [file, setFile] = useState<File | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<JobStatus>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [isHistoryLoading, setIsHistoryLoading] = useState(false);
  const [lookupId, setLookupId] = useState("");
  const [isLookupLoading, setIsLookupLoading] = useState(false);
  const [copyFeedback, setCopyFeedback] = useState<string | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const copyTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const analysisAnchorRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
      if (copyTimeoutRef.current) {
        clearTimeout(copyTimeoutRef.current);
      }
    };
  }, []);

  const clearPolling = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  }, []);

  useEffect(() => {
    if (copyTimeoutRef.current) {
      clearTimeout(copyTimeoutRef.current);
      copyTimeoutRef.current = null;
    }
    setCopyFeedback(null);
  }, [jobId]);

  const getErrorMessage = useCallback((err: unknown, fallback: string) => {
    if (axios.isAxiosError(err)) {
      const responseData = err.response?.data;
      if (typeof responseData === "string") {
        return responseData;
      }
      if (responseData && typeof responseData === "object" && "message" in responseData) {
        const message = (responseData as { message?: unknown }).message;
        if (typeof message === "string") {
          return message;
        }
      }
      return err.message || fallback;
    }
    return fallback;
  }, []);

  const fetchHistory = useCallback(
    async (options?: { limit?: number }) => {
      try {
        setHistoryError(null);
        setIsHistoryLoading(true);
        const res = await api.get("/api/v1/history", {
          params: { limit: options?.limit ?? 10 },
        });
        const items = Array.isArray(res.data) ? (res.data as HistoryItem[]) : [];
        setHistory(items);
      } catch (err) {
        setHistoryError(getErrorMessage(err, "Не удалось загрузить историю анализов."));
      } finally {
        setIsHistoryLoading(false);
      }
    },
    [getErrorMessage]
  );

  useEffect(() => {
    fetchHistory();
  }, [fetchHistory]);

  const applyResultPayload = useCallback(
    (payload: (PredictionResult & { status?: string }) | null | undefined, id: string, options?: { pendingMessage?: string }) => {
      if (!payload) {
        return false;
      }

      if (typeof payload.status === "string" && payload.status !== "done") {
        setStatus(normalizeStatus(payload.status));
        setResult(null);
        setJobId(id);
        setError(options?.pendingMessage ?? "Задача ещё выполняется. Попробуйте позже.");
        return false;
      }

      const { status: _ignored, ...rest } = payload as PredictionResult & { status?: string };
      setResult(rest as PredictionResult);
      setStatus("done");
      setJobId(id);
      setError(null);
      return true;
    },
    []
  );

  const fetchResultById = useCallback(
    async (id: string) => {
      const trimmed = id.trim();
      if (!trimmed) {
        setError("Укажите ID задачи для поиска.");
        return;
      }

      clearPolling();
      setIsSubmitting(false);
      setIsLookupLoading(true);
      try {
        const response = await api.get(`/api/v1/result/${trimmed}`);
        const success = applyResultPayload(response.data as PredictionResult & { status?: string }, trimmed);
        if (success) {
          fetchHistory();
        }
      } catch (err) {
        setError(getErrorMessage(err, "Не удалось получить результат по указанному ID."));
      } finally {
        setIsLookupLoading(false);
      }
    },
    [applyResultPayload, clearPolling, fetchHistory, getErrorMessage]
  );

  const handleLookupSubmit = useCallback(
    async (event: React.FormEvent<HTMLFormElement>) => {
      event.preventDefault();
      await fetchResultById(lookupId);
    },
    [fetchResultById, lookupId]
  );

  const handleHistorySelect = useCallback(
    async (id: string) => {
      setLookupId(id);
      await fetchResultById(id);
      if (analysisAnchorRef.current) {
        analysisAnchorRef.current.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    },
    [fetchResultById]
  );

  const handleCopyJobId = useCallback(async () => {
    if (!jobId) {
      return;
    }

    const setMessage = (message: string) => {
      setCopyFeedback(message);
      if (copyTimeoutRef.current) {
        clearTimeout(copyTimeoutRef.current);
      }
      copyTimeoutRef.current = setTimeout(() => {
        setCopyFeedback(null);
        copyTimeoutRef.current = null;
      }, 2000);
    };

    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(jobId);
        setMessage("ID скопирован");
        return;
      }

      const textarea = document.createElement("textarea");
      textarea.value = jobId;
      textarea.setAttribute("readonly", "");
      textarea.style.position = "absolute";
      textarea.style.left = "-9999px";
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand("copy");
      document.body.removeChild(textarea);
      setMessage("ID скопирован");
    } catch (err) {
      console.error("Failed to copy job id", err);
      setMessage("Не удалось скопировать");
    }
  }, [jobId]);

  const upload = async (): Promise<string | null> => {
    if (!file) {
      setError("Пожалуйста, выберите изображение растения для анализа.");
      return null;
    }

    const fd = new FormData();
    fd.append("file", file);

    try {
      const res = await api.post("/api/v1/upload", fd, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      return res.data.file_id as string;
    } catch (err) {
      setError(getErrorMessage(err, "Не удалось загрузить изображение."));
      return null;
    }
  };

  const poll = (jid: string) => {
    intervalRef.current = setInterval(async () => {
      try {
        const s = await api.get(`/api/v1/status/${jid}`);
        const nextStatus = normalizeStatus(s.data.status);
        setStatus(nextStatus);

        if (nextStatus === "done") {
          clearPolling();
          const r = await api.get(`/api/v1/result/${jid}`);
          applyResultPayload(r.data as PredictionResult & { status?: string }, jid);
          fetchHistory();
          setIsSubmitting(false);
        }

        if (nextStatus === "error") {
          clearPolling();
          setStatus("error");
          setError("Произошла ошибка при обработке задачи. Попробуйте еще раз.");
          setIsSubmitting(false);
        }
      } catch (err) {
        clearPolling();
        setError(getErrorMessage(err, "Не удалось получить статус задачи."));
        setIsSubmitting(false);
      }
    }, 1000);
  };

  const handleRun = async () => {
    setError(null);
    setResult(null);
    setStatus(null);
    clearPolling();
    setIsSubmitting(true);

    const fileId = await upload();
    if (!fileId) {
      setIsSubmitting(false);
      return;
    }

    try {
      const job = await api.post("/api/v1/predict", { file_id: fileId, model_version: "v1" });
      setJobId(job.data.job_id);
      setStatus("queued");
      poll(job.data.job_id);
    } catch (err) {
      setError(getErrorMessage(err, "Не удалось запустить анализ."));
      setIsSubmitting(false);
    }
  };

  const renderStatus = () => {
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
              onClick={handleCopyJobId}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") {
                  event.preventDefault();
                  handleCopyJobId();
                }
              }}
            >
              {jobId}
            </span>
            <button type="button" className="status__copy-btn" onClick={handleCopyJobId}>
              Скопировать
            </button>
            {copyFeedback && <span className="status__copy-feedback">{copyFeedback}</span>}
          </div>
        )}
      </div>
    );
  };

  const getConfidenceTone = (confidence: number): ConfidenceTone => {
    if (confidence >= 0.8) {
      return "high";
    }

    if (confidence >= CONFIDENCE_WARNING_THRESHOLD) {
      return "medium";
    }

    return "low";
  };

  const formatDate = (iso?: string | null) => {
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
  };

  return (
    <div className="card" ref={analysisAnchorRef}>
      <div className="field">
        <label htmlFor="image-upload" className="field__label">
          Изображение растения
        </label>
        <input
          id="image-upload"
          className="field__input"
          type="file"
          accept="image/*"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
        />
        <p className="field__hint">Поддерживаются изображения в форматах JPG, PNG или JPEG.</p>
      </div>

      <div className="actions">
        <button className="button" onClick={handleRun} disabled={isSubmitting || !file}>
          {isSubmitting ? "Анализируем..." : "Анализировать"}
        </button>
        <p className="actions__info">
          После загрузки мы будем периодически проверять статус задачи и покажем результат сразу, как он появится.
        </p>
      </div>

      {error && <div className="alert alert--error">{error}</div>}

      {renderStatus()}

      {result && (
        <div className="result">
          <h2 className="result__title">Результат анализа</h2>
          {result.confidence < CONFIDENCE_WARNING_THRESHOLD && (
            <div className="alert alert--warning">
              Уверенность модели ниже {Math.round(CONFIDENCE_WARNING_THRESHOLD * 100)}%. Проверьте качество
              изображения и попробуйте сделать новый снимок листа под лучшим освещением.
            </div>
          )}
          <dl className="result__grid">
            <div className="result__grid-item">
              <dt>Растение</dt>
              <dd>{result.plant}</dd>
            </div>
            <div className="result__grid-item">
              <dt>Заболевание</dt>
              <dd>{result.disease}</dd>
            </div>
            <div
              className={`result__grid-item result__grid-item--confidence result__grid-item--confidence-${getConfidenceTone(
                result.confidence
              )}`}
            >
              <dt>Уверенность модели</dt>
              <dd>{(result.confidence * 100).toFixed(1)}%</dd>
            </div>
            {result.created_at && (
              <div className="result__grid-item">
                <dt>Дата анализа</dt>
                <dd>{formatDate(result.created_at)}</dd>
              </div>
            )}
          </dl>

          {result.gradcam_url && (
            <figure className="result__figure">
              <img src={`${API_BASE_URL}${result.gradcam_url}`} alt="Grad-CAM visualization" />
              <figcaption>Тепловая карта уязвимых участков листа.</figcaption>
            </figure>
          )}

          {result.description && (
            <section className="result__section">
              <h3>Описание</h3>
              <p>{result.description}</p>
            </section>
          )}

          {result.pathogen && (
            <section className="result__section">
              <h3>Возбудитель</h3>
              <p>{result.pathogen}</p>
            </section>
          )}

          {result.treatment && (
            <section className="result__section">
              <h3>Рекомендации по лечению</h3>
              <p>{result.treatment}</p>
            </section>
          )}

          {result.prevention && (
            <section className="result__section">
              <h3>Профилактика</h3>
              <p>{result.prevention}</p>
            </section>
          )}
        </div>
      )}

      <section className="history">
        <div className="history__header">
          <h2 className="history__title">История анализов</h2>
          <button
            type="button"
            className="button button--ghost"
            onClick={() => fetchHistory()}
            disabled={isHistoryLoading}
          >
            {isHistoryLoading ? "Обновляем..." : "Обновить"}
          </button>
        </div>

        <form className="history__lookup" onSubmit={handleLookupSubmit}>
          <label className="history__label" htmlFor="history-lookup">
            Найти результат по ID задачи
          </label>
          <div className="history__lookup-controls">
            <input
              id="history-lookup"
              type="text"
              value={lookupId}
              onChange={(event) => setLookupId(event.target.value)}
              placeholder="Введите ID задачи"
            />
            <button type="submit" className="button" disabled={isLookupLoading}>
              {isLookupLoading ? "Ищем..." : "Найти"}
            </button>
          </div>
          <p className="history__hint">Введите сохранённый идентификатор задачи, чтобы повторно открыть результат.</p>
        </form>

        {historyError && <div className="alert alert--error">{historyError}</div>}

        {isHistoryLoading && <p className="history__loading">Загружаем историю...</p>}

        {history.length === 0 && !isHistoryLoading ? (
          <p className="history__empty">История пока пуста — выполните анализ, чтобы увидеть сохранённые результаты.</p>
        ) : (
          <ul className="history__list">
            {history.map((item) => (
              <li key={item.job_id} className="history__item">
                <div className="history__item-main">
                  <span className="history__job">{item.job_id}</span>
                  {item.created_at && (
                    <span className="history__meta">{formatDate(item.created_at)}</span>
                  )}
                </div>
                <div className="history__item-details">
                  <span className="history__plant">{item.plant ?? "Неизвестно"}</span>
                  <span
                    className={`history__confidence history__confidence--${getConfidenceTone(item.confidence ?? 0)}`}
                  >
                    {item.confidence != null ? `${(item.confidence * 100).toFixed(1)}%` : "—"}
                  </span>
                </div>
                <button
                  type="button"
                  className="button button--ghost"
                  onClick={() => {
                    void handleHistorySelect(item.job_id);
                  }}
                  disabled={isLookupLoading}
                >
                  Открыть
                </button>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}
