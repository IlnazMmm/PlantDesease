import React, { useCallback, useEffect, useRef } from "react";

import { HistoryPanel } from "../components/HistoryPanel";
import { ResultPanel } from "../components/ResultPanel";
import { StatusBanner } from "../components/StatusBanner";
import { UploadSection } from "../components/UploadSection";
import { useHistory } from "../hooks/useHistory";
import { usePrediction } from "../hooks/usePrediction";

export default function Home(): JSX.Element {
  const refreshHistoryRef = useRef<(() => void) | undefined>();

  const prediction = usePrediction({
    onResultLoaded: () => {
      if (refreshHistoryRef.current) {
        refreshHistoryRef.current();
      }
    },
  });

  const history = useHistory({
    loadResult: prediction.actions.loadResult,
    analysisAnchorRef: prediction.refs.analysisAnchorRef,
  });

  useEffect(() => {
    refreshHistoryRef.current = () => {
      void history.actions.refreshHistory();
    };
  }, [history.actions.refreshHistory]);

  const {
    state: { file, jobId, status, result, error, isSubmitting, copyFeedback },
    actions: { setFile, startPrediction, copyJobId },
    derived: { gradcamSrc },
    refs: { analysisAnchorRef },
  } = prediction;

  const {
    state: { history: historyItems, historyError, isHistoryLoading, lookupId, isLookupLoading },
    actions: { setLookupId, refreshHistory, lookupById, openFromHistory },
  } = history;

  const handleLookupSubmit = useCallback(
    (event: React.FormEvent<HTMLFormElement>) => {
      event.preventDefault();
      void lookupById();
    },
    [lookupById]
  );

  const handleOpenHistory = useCallback(
    (id: string) => {
      void openFromHistory(id);
    },
    [openFromHistory]
  );

  return (
    <div className="card" ref={analysisAnchorRef}>
      <UploadSection file={file} onFileChange={setFile} onSubmit={startPrediction} isSubmitting={isSubmitting} />

      {error && <div className="alert alert--error">{error}</div>}

      <StatusBanner status={status} jobId={jobId} onCopy={copyJobId} copyFeedback={copyFeedback} />

      <ResultPanel result={result} gradcamSrc={gradcamSrc} />

      <HistoryPanel
        history={historyItems}
        historyError={historyError}
        isHistoryLoading={isHistoryLoading}
        lookupId={lookupId}
        onLookupIdChange={setLookupId}
        onLookupSubmit={handleLookupSubmit}
        onRefresh={refreshHistory}
        onOpen={handleOpenHistory}
        isLookupLoading={isLookupLoading}
      />
    </div>
  );
}
