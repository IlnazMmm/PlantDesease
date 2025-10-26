import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import api, { API_BASE_URL } from "../services/api";
import { NullableJobStatus, PredictionResult } from "../types/prediction";
import { getErrorMessage } from "../utils/errors";
import { normalizeStatus } from "../utils/status";

interface UsePredictionOptions {
  onResultLoaded?: () => void;
}

interface ApplyResultOptions {
  pendingMessage?: string;
}

type ResultResponse = PredictionResult & { status?: string };

export function usePrediction(options?: UsePredictionOptions) {
  const onResultLoadedRef = useRef<(() => void) | undefined>(options?.onResultLoaded);
  useEffect(() => {
    onResultLoadedRef.current = options?.onResultLoaded;
  }, [options?.onResultLoaded]);

  const [file, setFile] = useState<File | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<NullableJobStatus>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
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

  const applyResultPayload = useCallback(
    (payload: ResultResponse | null | undefined, id: string, options?: ApplyResultOptions) => {
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
      setResult(rest);
      setStatus("done");
      setJobId(id);
      setError(null);
      onResultLoadedRef.current?.();
      return true;
    },
    []
  );

  const fetchResultById = useCallback(
    async (id: string, options?: ApplyResultOptions) => {
      const trimmed = id.trim();
      if (!trimmed) {
        setError("Укажите ID задачи для поиска.");
        return false;
      }

      clearPolling();
      setResult(null);
      setStatus(null);
      setIsSubmitting(false);

      try {
        const response = await api.get(`/api/v1/result/${trimmed}`);
        return applyResultPayload(response.data as ResultResponse, trimmed, options);
      } catch (err) {
        setError(getErrorMessage(err, "Не удалось получить результат по указанному ID."));
        return false;
      }
    },
    [applyResultPayload, clearPolling]
  );

  const uploadFile = useCallback(async () => {
    if (!file) {
      setError("Пожалуйста, выберите изображение растения для анализа.");
      return null;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await api.post("/api/v1/upload", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      return response.data.file_id as string;
    } catch (err) {
      setError(getErrorMessage(err, "Не удалось загрузить изображение."));
      return null;
    }
  }, [file]);

  const pollStatus = useCallback(
    (jid: string) => {
      intervalRef.current = setInterval(async () => {
        try {
          const statusResponse = await api.get(`/api/v1/status/${jid}`);
          const nextStatus = normalizeStatus(statusResponse.data.status);
          setStatus(nextStatus);

          if (nextStatus === "done") {
            clearPolling();
            const resultResponse = await api.get(`/api/v1/result/${jid}`);
            const success = applyResultPayload(resultResponse.data as ResultResponse, jid);
            if (success) {
              setIsSubmitting(false);
            }
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
    },
    [applyResultPayload, clearPolling]
  );

  const startPrediction = useCallback(async () => {
    setError(null);
    setResult(null);
    setStatus(null);
    clearPolling();
    setIsSubmitting(true);

    const fileId = await uploadFile();
    if (!fileId) {
      setIsSubmitting(false);
      return;
    }

    try {
      const job = await api.post("/api/v1/predict", { file_id: fileId, model_version: "v1" });
      setJobId(job.data.job_id);
      setStatus("queued");
      pollStatus(job.data.job_id);
    } catch (err) {
      setError(getErrorMessage(err, "Не удалось запустить анализ."));
      setIsSubmitting(false);
    }
  }, [clearPolling, pollStatus, uploadFile]);

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
      // eslint-disable-next-line no-console
      console.error("Failed to copy job id", err);
      setMessage("Не удалось скопировать");
    }
  }, [jobId]);

  const gradcamSrc = useMemo(() => {
    if (!result?.gradcam_url) {
      return null;
    }

    const cacheKey = encodeURIComponent(result.job_id ?? result.gradcam_url);
    const separator = result.gradcam_url.includes("?") ? "&" : "?";
    return `${API_BASE_URL}${result.gradcam_url}${separator}v=${cacheKey}`;
  }, [result?.gradcam_url, result?.job_id]);

  return {
    state: {
      file,
      jobId,
      status,
      result,
      error,
      isSubmitting,
      copyFeedback,
    },
    actions: {
      setFile,
      startPrediction,
      loadResult: fetchResultById,
      copyJobId: handleCopyJobId,
    },
    derived: {
      gradcamSrc,
    },
    refs: {
      analysisAnchorRef,
    },
  };
}
