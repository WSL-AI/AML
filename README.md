# AML

Учебный проект по поиску отмывания денег (anti-money laundering) в транзакциях. Данные синтетические, это IBM AML dataset с Kaggle. Основное обучение строится на `HI-Small_Trans.csv` (5 млн транзакций, отмывания около 0.1%).

## Как устроено

![схема ETL](docs/images/etl.png)

Сначала ETL читает CSV, проверяет его, режет по времени на четыре части (train, threshold, val, test) и считает признаки. Категории кодируются через ordered target encoding (OTE). Encoder обучается только на train и сохраняется вместе с данными, чтобы не было утечки из будущего.

Дальше всё делается в notebooks: сравнение моделей, подбор параметров через Optuna и выбор порога.

Подробнее: [про ETL](docs/ETL.md), [про датасет](docs/dataset.md).

## Результаты (HI-Small)

Пробовал логистическую регрессию, CatBoost, LightGBM и XGBoost, у бустингов по 25 trials Optuna. На validation (7 сентября) три бустинга почти одинаковые, логрегрессия заметно хуже:

![AP моделей на validation](docs/images/models_ap.png)

XGBoost первый, но бутстрап показал, что его отрыв от LightGBM (+0.005 AP) в пределах шума. Так что выбран он скорее формально.

На test (9–10 сентября, модель его не видела) у XGBoost вышло так:

| Метрика | Значение |
|---|---:|
| Average Precision | 0.604 |
| Precision | 78.4% |
| Recall | 51.7% |
| Алертов | 630 |
| Из них правда отмывание | 494 из 956 |

Примерно 4 из 5 алертов попадают в цель, и находится половина всех случаев отмывания. Но есть большая проблема: **все алерты пришлись на платежи ACH**. В других форматах (Cheque, Credit Card, Cash, Bitcoin) было 122 случая отмывания, и модель не нашла ни одного. В train 83% отмывания было на ACH, и модель выучила этот перекос.

Ещё я проверил XGBoost и LightGBM на трёх днях подряд (backtest). Тут алерты считаются по-другому: каждый день берётся топ 0.1% транзакций, поэтому с таблицей выше цифры сравнивать нельзя.

![backtest по дням](docs/images/backtest.png)

XGBoost выигрывает все три дня: AP в среднем 0.55 против 0.26, recall 53% против 40%. Но в backtest параметры фиксированные, и LightGBM с ними, похоже, переобучается: после подбора в notebook 03 он почти догнал XGBoost. Так что это скорее про настройки, чем про алгоритм.

Подробные выводы лежат в конце каждого notebook. В [reports/](reports/full/RESULTS.md) остался старый прогон на LI-Small (там test AP 0.374, recall 32%).

## Как запустить

Нужен Python 3.13+ и [uv](https://docs.astral.sh/uv/). Все команды запускаются из корня репозитория.

```bash
uv sync --locked --extra dev
```

### Данные

Скачиваются через Kaggle CLI:

```bash
uv tool install kaggle
kaggle auth login
```

Скрипт пока умеет качать только `LI-Small_Trans.csv` и кладёт его в `data/raw/`:

```bash
python3 scripts/download_dataset.py
```

`HI-Small_Trans.csv` можно скачать так же через Kaggle CLI (если придёт zip, распакуйте его в `data/raw/`):

```bash
kaggle datasets download -d ealtman2019/ibm-transactions-for-anti-money-laundering-aml -f HI-Small_Trans.csv -p data/raw
```

Или руками с [Kaggle](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml/data). Про другие способы авторизации (например API token) написал в [docs/ETL.md](docs/ETL.md#откуда-взять-данные).

### ETL

Сначала лучше прогнать на маленьком куске, чтобы проверить, что всё работает:

```bash
uv run --locked python -m aml.cli.etl \
  --source data/raw/HI-Small_Trans.csv \
  --output artifacts/runs-smoke \
  --smoke
```

`--smoke` берёт каждый 50-й счёт отправителя со всей его историей. Папка отдельная, чтобы не испортить основной запуск.

Полный запуск (то же самое делает `notebooks/00_etl.ipynb`):

```bash
uv run --locked python -m aml.cli.etl \
  --source data/raw/HI-Small_Trans.csv \
  --output artifacts/runs
```

В конце команда печатает путь к новому run. Там лежат данные, timestamps и json с описанием источника и признаков. Если появился файл `SUCCESS`, значит всё записалось до конца. `active.json` указывает на последний удачный run.

После этого можно сразу открывать notebook 03 и обучать модели. Notebook 02 не обязателен, там я сравнивал разные способы кодировать категории.

### Notebooks и тесты

```bash
uv run --locked --extra dev python -m pytest -q
uv run --locked jupyter lab
```

Notebooks берут run из `artifacts/runs/active.json`. Чтобы взять другой, задайте `AML_RUN_DIR=artifacts/runs/<run_id>`. Данных и весов моделей в репозитории нет, так что после clone сначала надо прогнать ETL.

| Notebook | Что там |
|---|---|
| `00_etl.ipynb` | запуск ETL и проверка, что всё сохранилось |
| `01_eda.ipynb` | смотрю на данные и распределения в train |
| `02_feature_engineering.ipynb` | сравнение CatBoost на сырых категориях и OTE |
| `03_training.ipynb` | обучение моделей, Optuna, бутстрап, выбор лучшей |
| `04_evaluation.ipynb` | выбор порога, проверка на test, ошибки по сегментам |
| `05_data_drift.ipynb` | как меняются признаки со временем |
| `06_temporal_backtesting.ipynb` | backtest на нескольких днях |

Каждый запуск сохраняется в свою папку: 02 создаёт новый набор данных, 03 пишет в `training/<run_id>`, 04 и 05 в `evaluations/<run_id>` и `drift/<run_id>`, backtest в `artifacts/backtests/`. Указатель на последний результат обновляется только если этап закончился без ошибок.

## Что где лежит

```text
scripts/download_dataset.py  # скачать CSV с Kaggle
src/aml/cli/etl.py           # запуск ETL
src/aml/data/                # чтение, проверка, split, сохранение
src/aml/features/            # признаки и OTE encoder
src/aml/model/               # обучение, метрики, backtest
src/aml/runs.py              # папки запусков и указатели
notebooks/                   # анализ, обучение, оценка
tests/                       # тесты
docs/                        # документация
reports/                     # старый прогон на LI-Small
```

`data/` и `artifacts/` в git не попадают, они только локально. Версии библиотек зафиксированы в `uv.lock`.
