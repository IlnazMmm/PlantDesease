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
};

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
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

  const upload = async () => {
    if (!file) {
      setError("Пожалуйста, выберите файл изображения.");
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
        setStatus(s.data.status);

        if (s.data.status === "done") {
          clearPolling();
          const r = await api.get(`/api/v1/result/${jid}`);
          setResult(r.data);
        }

        if (s.data.status === "error") {
          clearPolling();
          setStatus("error");
          setError("Произошла ошибка при обработке задачи.");
        }
      } catch (err) {
        clearPolling();
        setError(getErrorMessage(err, "Не удалось получить статус задачи."));
      }
    }, 1000);
  };

  const handleRun = async () => {
    setError(null);
    clearPolling();

    const file_id = await upload();
    if (!file_id) {
      return;
    }

    try {
      const job = await api.post("/api/v1/predict", { file_id, model_version: "v1" });
      setJobId(job.data.job_id);
      setStatus("queued");
      setResult(null);
      poll(job.data.job_id);
    } catch (err) {
      setError(getErrorMessage(err, "Не удалось запустить анализ."));
    }
  };

  return (
    <div className="bg-white p-6 rounded shadow">
      <div className="mb-4">
        <input
          type="file"
          accept="image/*"
          onChange={(e) => setFile(e.target.files?.[0] || null)}
        />
      </div>
      <div className="mb-4">
        <button className="bg-blue-600 text-white px-4 py-2 rounded" onClick={handleRun}>
          Analyze
        </button>
      </div>

      {error && <div className="mb-4 text-red-600">{error}</div>}

      {jobId && (
        <div className="mb-4">
          Job: {jobId} — Status: {status}
        </div>
      )}

      {result && (
        <div className="mt-4">
          <h2 className="text-xl font-semibold">Result</h2>
          <p>
            <strong>Plant:</strong> {result.plant}
          </p>
          <p>
            <strong>Disease:</strong> {result.disease}
          </p>
          <p>
            <strong>Confidence:</strong> {(result.confidence * 100).toFixed(1)}%
          </p>
          {result.gradcam_url && (
            <img
              src={`${API_BASE_URL}${result.gradcam_url}`}
              alt="gradcam"
              className="mt-2 max-w-full rounded shadow"
            />
          )}
          {result.description && (
            <>
              <h3 className="mt-2 font-semibold">Description</h3>
              <p>{result.description}</p>
            </>
          )}
          {result.treatment && (
            <>
              <h3 className="mt-2 font-semibold">Treatment</h3>
              <p>{result.treatment}</p>
            </>
          )}
        </div>
      )}
    </div>
  );
}
