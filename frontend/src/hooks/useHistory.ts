import { useCallback, useEffect, useState } from "react";

import api from "../services/api";
import { HistoryItem } from "../types/prediction";
import { getErrorMessage } from "../utils/errors";

interface UseHistoryOptions {
  loadResult: (id: string, options?: { pendingMessage?: string }) => Promise<boolean>;
  analysisAnchorRef: React.RefObject<HTMLDivElement>;
}

export function useHistory({ loadResult, analysisAnchorRef }: UseHistoryOptions) {
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [isHistoryLoading, setIsHistoryLoading] = useState(false);
  const [lookupId, setLookupId] = useState("");
  const [isLookupLoading, setIsLookupLoading] = useState(false);

  const fetchHistory = useCallback(
    async (options?: { limit?: number }) => {
      try {
        setHistoryError(null);
        setIsHistoryLoading(true);
        const response = await api.get("/api/v1/history", {
          params: { limit: options?.limit ?? 10 },
        });
        const items = Array.isArray(response.data) ? (response.data as HistoryItem[]) : [];
        setHistory(items);
      } catch (err) {
        setHistoryError(getErrorMessage(err, "Не удалось загрузить историю анализов."));
      } finally {
        setIsHistoryLoading(false);
      }
    },
    []
  );

  useEffect(() => {
    fetchHistory();
  }, [fetchHistory]);

  const lookupById = useCallback(async () => {
    const trimmed = lookupId.trim();
    setIsLookupLoading(true);
    try {
      const success = await loadResult(trimmed);
      if (success) {
        await fetchHistory();
        if (analysisAnchorRef.current) {
          analysisAnchorRef.current.scrollIntoView({ behavior: "smooth", block: "start" });
        }
      }
      return success;
    } finally {
      setIsLookupLoading(false);
    }
  }, [analysisAnchorRef, fetchHistory, loadResult, lookupId]);

  const openFromHistory = useCallback(
    async (id: string) => {
      setLookupId(id);
      setIsLookupLoading(true);
      try {
        const success = await loadResult(id);
        if (success && analysisAnchorRef.current) {
          analysisAnchorRef.current.scrollIntoView({ behavior: "smooth", block: "start" });
        }
        return success;
      } finally {
        setIsLookupLoading(false);
      }
    },
    [analysisAnchorRef, loadResult]
  );

  return {
    state: {
      history,
      historyError,
      isHistoryLoading,
      lookupId,
      isLookupLoading,
    },
    actions: {
      setLookupId,
      refreshHistory: fetchHistory,
      lookupById,
      openFromHistory,
    },
  };
}
