import React from "react";

import { HistoryItem } from "../types/prediction";
import { formatConfidence, formatDateTime, getConfidenceTone } from "../utils/prediction";

interface HistoryPanelProps {
  history: HistoryItem[];
  historyError: string | null;
  isHistoryLoading: boolean;
  lookupId: string;
  onLookupIdChange: (value: string) => void;
  onLookupSubmit: (event: React.FormEvent<HTMLFormElement>) => void;
  onRefresh: () => void | Promise<void>;
  onOpen: (jobId: string) => void | Promise<void>;
  isLookupLoading: boolean;
}

export function HistoryPanel({
  history,
  historyError,
  isHistoryLoading,
  lookupId,
  onLookupIdChange,
  onLookupSubmit,
  onRefresh,
  onOpen,
  isLookupLoading,
}: HistoryPanelProps) {
  return (
    <section className="history">
      <div className="history__header">
        <h2 className="history__title">История анализов</h2>
        <button type="button" className="button button--ghost" onClick={onRefresh} disabled={isHistoryLoading}>
          {isHistoryLoading ? "Обновляем..." : "Обновить"}
        </button>
      </div>

      <form className="history__lookup" onSubmit={onLookupSubmit}>
        <label className="history__label" htmlFor="history-lookup">
          Найти результат по ID задачи
        </label>
        <div className="history__lookup-controls">
          <input
            id="history-lookup"
            type="text"
            value={lookupId}
            onChange={(event) => onLookupIdChange(event.target.value)}
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
                {item.created_at && <span className="history__meta">{formatDateTime(item.created_at)}</span>}
              </div>
              <div className="history__item-details">
                <span className="history__plant">{item.plant ?? "Неизвестно"}</span>
                <span className={`history__confidence history__confidence--${getConfidenceTone(item.confidence ?? 0)}`}>
                  {formatConfidence(item.confidence)}
                </span>
              </div>
              <button
                type="button"
                className="button button--ghost"
                onClick={() => onOpen(item.job_id)}
                disabled={isLookupLoading}
              >
                Открыть
              </button>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
