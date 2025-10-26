import React from "react";

import { PredictionResult } from "../types/prediction";
import { CONFIDENCE_WARNING_THRESHOLD, getConfidenceTone } from "../utils/prediction";

interface ResultPanelProps {
  result: PredictionResult | null;
  gradcamSrc: string | null;
}

export function ResultPanel({ result, gradcamSrc }: ResultPanelProps) {
  if (!result) {
    return null;
  }

  return (
    <section className="result">
      <h2 className="result__title">Результат анализа</h2>
      {result.confidence < CONFIDENCE_WARNING_THRESHOLD && (
        <div className="alert alert--warning">
          Уверенность модели ниже {Math.round(CONFIDENCE_WARNING_THRESHOLD * 100)}%. Проверьте качество изображения и
          попробуйте сделать новый снимок листа под лучшим освещением.
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
      </dl>

      {gradcamSrc && (
        <figure className="result__figure">
          <img key={result.job_id ?? result.gradcam_url} src={gradcamSrc} alt="Grad-CAM visualization" />
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
    </section>
  );
}
