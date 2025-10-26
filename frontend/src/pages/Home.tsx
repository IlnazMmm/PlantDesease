import React, { useEffect, useRef, useState } from "react";
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
};

type JobStatus = "queued" | "processing" | "done" | "error" | "saved" | null;

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
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, []);

  const clearPolling = () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  };

  const getErrorMessage = (err: unknown, fallback: string) => {
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
  };

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
          setResult(r.data);
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
        {jobId && <span className="status__job">ID задачи: {jobId}</span>}
      </div>
    );
  };

  return (
    <div className="card">
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
          <dl className="result__grid">
            <div>
              <dt>Растение</dt>
              <dd>{result.plant}</dd>
            </div>
            <div>
              <dt>Заболевание</dt>
              <dd>{result.disease}</dd>
            </div>
            <div>
              <dt>Уверенность модели</dt>
              <dd>{(result.confidence * 100).toFixed(1)}%</dd>
            </div>
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
    </div>
  );
}
