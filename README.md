# Plant Disease Detector

Комплексное решение для детекции заболеваний растений по фотографиям листьев. Проект объединяет модель компьютерного зрения, FastAPI-бэкенд с асинхронной обработкой заданий, React-интерфейс и вспомогательные инструменты для обучения модели.

## Основные возможности

- Загрузка изображения растения и постановка задачи на анализ через REST API или веб-интерфейс.【F:backend/app/main.py†L35-L109】【F:frontend/src/pages/Home.tsx†L60-L79】
- Асинхронная обработка предсказаний с отображением прогресса, историей анализов и возможностью повторно открыть результат по ID задачи.【F:backend/app/services/job_store.py†L9-L74】【F:frontend/src/hooks/usePrediction.ts†L61-L260】【F:frontend/src/components/HistoryPanel.tsx†L30-L91】
- Сохранение результатов, Grad-CAM тепловых карт и пользовательского фидбэка в базе данных для последующего аудита.【F:backend/app/main.py†L60-L193】【F:backend/app/models/db_models.py†L8-L35】
- Обогащение предсказаний экспертными описаниями заболеваний, советами по лечению и профилактике из встроенной базы знаний.【F:backend/app/models/knowledge_base.py†L11-L81】
- Docker Compose-окружение с автоматическими миграциями БД и Nginx-прокси, проксирующим запросы к фронтенду и API.【F:docker-compose.yml†L3-L77】【F:nginx.conf†L3-L16】

## Структура репозитория

- `backend/` — FastAPI-приложение, фоновые задания и доступ к БД; Dockerfile собирает сервис с установленным ML-пакетом из `dist/`.【F:backend/Dockerfile†L1-L17】
- `frontend/` — React 18 + TypeScript SPA с хуками для работы с API и визуализацией Grad-CAM.【F:frontend/package.json†L6-L19】【F:frontend/src/components/ResultPanel.tsx†L16-L78】
- `ml/` — код обучения и инференса модели EfficientNet-B0, скрипты подготовки датасета и упаковки весов.【F:ml/src/train.py†L1-L150】【F:ml/src/infer.py†L1-L41】【F:ml/data/split_dataset.py†L1-L40】
- `docker-compose.yml`, `nginx.conf` — оркестрация сервисов и обратный прокси.【F:docker-compose.yml†L3-L74】【F:nginx.conf†L3-L16】

## Быстрый старт (Docker Compose)

1. Создайте файл `.env` в корне проекта и задайте параметры Postgres:
   ```env
   POSTGRES_HOST=db
   POSTGRES_PORT=5432
   POSTGRES_DB=plant_disease
   POSTGRES_USER=plant_app
   POSTGRES_PASSWORD=secret
   ```
2. Соберите и запустите окружение:
   ```bash
   docker compose up --build
   ```
   Compose поднимет Postgres, выполнит миграции (`python -m app.migrate`) и развернёт бэкенд, фронтенд и Nginx-прокси.【F:docker-compose.yml†L22-L74】
3. Откройте `http://localhost` — веб-интерфейс доступен через Nginx, а API проксируется по пути `/api/`。【F:nginx.conf†L3-L16】

## Локальный запуск без Docker

### Бэкенд
1. Установите Python 3.10+ и создайте виртуальное окружение.
2. Установите зависимости:
   ```bash
   pip install -r backend/requirements.txt
   ```
   Файл включает локальный wheel `dist/ml-1.0.0-py3-none-any.whl` с кодом инференса и весами модели.【F:backend/requirements.txt†L1-L15】
3. Настройте доступ к базе Postgres через переменные окружения (`POSTGRES_HOST`, `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`); при отсутствии переменных используется строка подключения по умолчанию `postgresql://postgres:postgres@localhost:5432/postgres`.【F:backend/app/db.py†L20-L32】
4. Выполните миграции:
   ```bash
   python -m app.migrate
   ```
   Скрипт создаст таблицы и выдаст права на схему `public`.【F:backend/app/migrate.py†L9-L23】
5. Запустите приложение:
   ```bash
   uvicorn app.main:app --reload
   ```

### Фронтенд
1. Перейдите в `frontend/` и установите зависимости:
   ```bash
   npm install
   ```
2. (Необязательно) Укажите URL API, если он отличается, через `REACT_APP_API_URL`.
   Базовое значение — `/api` (относительный путь).【F:frontend/src/services/api.ts†L3-L7】
3. Запустите дев-сервер:
   ```bash
   npm start
   ```
   Приложение откроется на `http://localhost:3000` и будет проксировать запросы к API.

## API

| Метод | Путь | Описание |
|-------|------|----------|
| `POST` | `/api/v1/upload` | Принимает файл изображения, сохраняет его и возвращает `file_id`.|【F:backend/app/main.py†L35-L46】
| `POST` | `/api/v1/predict` | Ставит задачу инференса по `file_id`, возвращает `job_id` и статус.|【F:backend/app/main.py†L48-L58】
| `GET` | `/api/v1/status/{job_id}` | Проверяет состояние задачи (`queued`, `processing`, `done`, `error`).|【F:backend/app/main.py†L101-L108】
| `GET` | `/api/v1/result/{job_id}` | Возвращает подробный результат с описанием болезни и Grad-CAM, либо промежуточный статус.|【F:backend/app/main.py†L109-L151】
| `GET` | `/api/v1/history?limit=N` | Список последних `N` задач с краткой информацией и ссылкой на Grad-CAM.|【F:backend/app/main.py†L154-L185】
| `POST` | `/api/v1/feedback` | Сохраняет корректирующий фидбэк пользователя (правильный класс).|【F:backend/app/main.py†L187-L193】
| `GET` | `/static/gradcam/{name}` | Выдаёт сохранённую тепловую карту Grad-CAM.|【F:backend/app/main.py†L195-L200】

## Интерфейс пользователя

Главная страница позволяет загрузить файл, отслеживать статус, просмотреть результат с доверительным уровнем, Grad-CAM-визуализацией и текстовыми рекомендациями, а также открыть историю последних анализов или найти результат по ID.【F:frontend/src/pages/Home.tsx†L60-L79】【F:frontend/src/components/UploadSection.tsx†L12-L33】【F:frontend/src/components/StatusBanner.tsx†L20-L47】【F:frontend/src/components/ResultPanel.tsx†L16-L78】【F:frontend/src/components/HistoryPanel.tsx†L30-L91】

## Обработка заданий и хранение данных

- Задания инференса регистрируются в потокобезопасном хранилище `JobStore`, которое хранит статус, результат и ошибку до завершения обработки.【F:backend/app/services/job_store.py†L9-L74】
- После завершения расчёта предсказания и Grad-CAM результат записывается в таблицу `results`, а исходный файл и тепловая карта сохраняются на диске.【F:backend/app/main.py†L60-L149】【F:backend/app/models/db_models.py†L15-L29】
- Репозиторий советов `AdviceRepository` подмешивает в ответ описание болезни, рекомендации по лечению и профилактике из JSON-файла `plant_advice.json` (значения по умолчанию заданы для неизвестных классов).【F:backend/app/models/knowledge_base.py†L22-L77】

## Модель и обучение

- За инференс отвечает обёртка `ModelPredictor`, использующая функции `load_model` и `predict` из пакета `ml` и веса `ml/models/model_v3.pth`. (Пакет устанавливается из локального wheel при установке зависимостей.)【F:backend/app/services/model_loader.py†L1-L37】
- Скрипт `ml/src/train.py` обучает EfficientNet-B0 на датасете ImageFolder с возможностью создания валидационного сплита, настройки гиперпараметров и сохранения лучшей модели вместе с историей обучения.【F:ml/src/train.py†L19-L150】
- Для инференса вне бэкенда можно использовать CLI из `ml/src/infer.py`, который грузит чекпойнт и возвращает класс с уверенностью.【F:ml/src/infer.py†L1-L41】
- В каталоге `ml/data` находятся вспомогательные скрипты для скачивания и разбиения датасета PlantVillage на `train/` и `valid/`.【F:ml/data/download_dataset.py†L1-L24】【F:ml/data/split_dataset.py†L1-L40】

## Тестирование

Бэкенд покрыт HTTP-интеграционными тестами на pytest (`backend/tests`). Они проверяют загрузку файлов, постановку и завершение задач, историю, Grad-CAM и обработку ошибок.
Запуск:
```bash
pytest backend/tests
```
【F:backend/tests/test_api.py†L1-L143】

## Полезные команды

- Запуск миграций вручную: `python -m app.migrate`.【F:backend/app/migrate.py†L9-L23】
- Запуск дев-сервера бэкенда: `uvicorn app.main:app --reload`.
- Пересборка образов Docker: `docker compose build`.
- Повторное обучение модели: `python -m ml.src.train --data <путь к данным> --out ml/models/model_v3.pth --device cuda` (при наличии GPU).【F:ml/src/train.py†L105-L150】

## Обратная связь и дальнейшее развитие

В результаты добавляется идентификатор задачи и ссылка на Grad-CAM, поэтому их можно повторно открыть через раздел истории. Таблица `feedback` хранит пользовательские корректировки меток, что позволяет в будущем дообучить модель на реальных данных фермеров.【F:backend/app/main.py†L109-L193】【F:backend/app/models/db_models.py†L31-L35】


## Экспертное подтверждение низкоуверенных результатов

Для повышения качества диагностики бэкенд сравнивает уверенность модели с порогом `CONFIDENCE_THRESHOLD`.
Значение по умолчанию — `0.70`; его можно переопределить через переменную окружения:

```env
CONFIDENCE_THRESHOLD=0.70
```

Если `confidence >= CONFIDENCE_THRESHOLD`, результат считается обычным и получает статус проверки `not_required`.
Если `confidence < CONFIDENCE_THRESHOLD`, запись результата помечается как требующая проверки агрономом:

- `review_required: true`;
- `review_status: "pending"`;
- в ответе `GET /api/v1/result/{job_id}` возвращается предупреждение `review_warning`.

Новые поля результата сохраняются в таблице `results` и возвращаются в деталях результата и истории:

- `review_required` — требуется ли экспертная проверка;
- `review_status` — один из `not_required`, `pending`, `confirmed`, `corrected`;
- `expert_label` — класс, выбранный экспертом, если он указан;
- `expert_comment` — комментарий эксперта;
- `reviewed_at` — дата и время проверки.

### Endpoint подтверждения

`POST /api/v1/review/{job_id}` сохраняет решение агронома и дополнительно пишет выбранную метку в существующую таблицу `feedback` для совместимости с прежним механизмом обратной связи.

Пример тела запроса:

```json
{
  "confirmed": true,
  "expert_label": "Tomato___Late_blight",
  "expert_comment": "Диагноз подтвержден агрономом"
}
```

Логика статусов:

- если `confirmed = true` и `expert_label` совпадает с предсказанным классом, сохраняется `review_status = "confirmed"`;
- если эксперт выбрал другой класс или отправил неподтвержденный диагноз, сохраняется `review_status = "corrected"`;
- комментарий, выбранная метка и `reviewed_at` сохраняются в результате анализа.

Во фронтенде для результатов с `review_required = true` показывается предупреждение «Уверенность модели ниже порога. Требуется подтверждение специалиста», форма подтверждения агрономом и статус проверки. В истории отображаются человекочитаемые статусы: «Подтверждение не требуется», «Ожидает подтверждения», «Подтверждено», «Скорректировано экспертом».
