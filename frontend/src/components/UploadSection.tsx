import React from "react";

interface UploadSectionProps {
  file: File | null;
  onFileChange: (file: File | null) => void;
  onSubmit: () => void | Promise<void>;
  isSubmitting: boolean;
}

export function UploadSection({ file, onFileChange, onSubmit, isSubmitting }: UploadSectionProps) {
  return (
    <section className="field">
      <label htmlFor="image-upload" className="field__label">
        Изображение растения
      </label>
      <input
        id="image-upload"
        className="field__input"
        type="file"
        accept="image/*"
        onChange={(event) => onFileChange(event.target.files?.[0] ?? null)}
      />
      <p className="field__hint">Поддерживаются изображения в форматах JPG, PNG или JPEG.</p>

      <div className="actions">
        <button className="button" onClick={onSubmit} disabled={isSubmitting || !file}>
          {isSubmitting ? "Анализируем..." : "Анализировать"}
        </button>
        <p className="actions__info">
          После загрузки мы будем периодически проверять статус задачи и покажем результат сразу, как он появится.
        </p>
      </div>
    </section>
  );
}
