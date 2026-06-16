import React, { useEffect, useMemo, useState } from "react";

import { PredictionResult, ReviewPayload } from "../types/prediction";
import { CONFIDENCE_WARNING_THRESHOLD, getConfidenceTone } from "../utils/prediction";
import { getReviewStatusLabel } from "../utils/review";

interface ResultPanelProps {
  result: PredictionResult | null;
  gradcamSrc: string | null;
  labels: string[];
  isReviewSaving: boolean;
  reviewSuccess: string | null;
  onSubmitReview: (payload: ReviewPayload) => void | Promise<boolean>;
}

export function ResultPanel({
  result,
  gradcamSrc,
  labels,
  isReviewSaving,
  reviewSuccess,
  onSubmitReview,
}: ResultPanelProps) {
  const [expertLabel, setExpertLabel] = useState("");
  const [expertComment, setExpertComment] = useState("");
  const [confirmed, setConfirmed] = useState(true);

  useEffect(() => {
    setExpertLabel(result?.expert_label ?? result?.label ?? "");
    setExpertComment(result?.expert_comment ?? "");
    setConfirmed(result?.review_status !== "corrected");
  }, [result?.job_id, result?.label, result?.expert_label, result?.expert_comment, result?.review_status]);

  const reviewStatusLabel = useMemo(() => getReviewStatusLabel(result?.review_status), [result?.review_status]);

  if (!result) {
    return null;
  }

  const submitReview = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    void onSubmitReview({
      confirmed,
      expert_label: expertLabel || result.label || null,
      expert_comment: expertComment || null,
    });
  };

  const confirmDiagnosis = () => {
    setConfirmed(true);
    setExpertLabel(result.label ?? expertLabel);
  };

  return (
    <section className="result">
      <h2 className="result__title">Результат анализа</h2>
      {result.review_required ? (
        <div className="alert alert--warning">
          {result.review_warning ?? "Уверенность модели ниже порога. Требуется подтверждение специалиста"}
        </div>
      ) : result.confidence < CONFIDENCE_WARNING_THRESHOLD ? (
        <div className="alert alert--warning">
          Уверенность модели ниже {Math.round(CONFIDENCE_WARNING_THRESHOLD * 100)}%. Проверьте качество изображения и
          попробуйте сделать новый снимок листа под лучшим освещением.
        </div>
      ) : null}
      {reviewSuccess && <div className="alert alert--success">{reviewSuccess}</div>}
      <dl className="result__grid">
        <div className="result__grid-item">
          <dt>Растение</dt>
          <dd>{result.plant}</dd>
        </div>
        <div className="result__grid-item">
          <dt>Заболевание</dt>
          <dd>{result.disease}</dd>
        </div>
        <div className="result__grid-item">
          <dt>Статус проверки</dt>
          <dd>{reviewStatusLabel}</dd>
        </div>
        <div
          className={`result__grid-item result__grid-item--confidence result__grid-item--confidence-${getConfidenceTone(
            result.confidence
          )}`}
        >
          <dt>Уверенность модели</dt>
          <dd>{(result.confidence * 100).toFixed(1)}%</dd>
        </div>
      </dl>

      {gradcamSrc && (
        <figure className="result__figure">
          <img key={result.job_id ?? result.gradcam_url} src={gradcamSrc} alt="Grad-CAM visualization" />
          <figcaption>Тепловая карта уязвимых участков листа.</figcaption>
        </figure>
      )}

      {result.review_required && (
        <form className="review-form" onSubmit={submitReview}>
          <h3>Подтверждение агрономом</h3>
          <button type="button" className="button button--ghost" onClick={confirmDiagnosis} disabled={isReviewSaving}>
            Подтвердить диагноз
          </button>
          <label className="field__label" htmlFor="expert-label">Корректный диагноз</label>
          <select
            id="expert-label"
            className="field__input"
            value={expertLabel}
            onChange={(event) => {
              setExpertLabel(event.target.value);
              setConfirmed(event.target.value === result.label);
            }}
          >
            <option value="">Выберите класс болезни</option>
            {(labels.length ? labels : [result.label].filter(Boolean) as string[]).map((label) => (
              <option key={label} value={label}>{label}</option>
            ))}
          </select>
          <label className="field__label" htmlFor="expert-comment">Комментарий</label>
          <textarea
            id="expert-comment"
            className="field__input review-form__comment"
            value={expertComment}
            onChange={(event) => setExpertComment(event.target.value)}
            placeholder="Добавьте комментарий специалиста"
          />
          <button type="submit" className="button" disabled={isReviewSaving || !expertLabel}>
            {isReviewSaving ? "Сохраняем..." : "Сохранить подтверждение"}
          </button>
        </form>
      )}

      {result.description && <section className="result__section"><h3>Описание</h3><p>{result.description}</p></section>}
      {result.pathogen && <section className="result__section"><h3>Возбудитель</h3><p>{result.pathogen}</p></section>}
      {result.treatment && <section className="result__section"><h3>Рекомендации по лечению</h3><p>{result.treatment}</p></section>}
      {result.prevention && <section className="result__section"><h3>Профилактика</h3><p>{result.prevention}</p></section>}
    </section>
  );
}
