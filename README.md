# SRS pipeline — первый запуск

Эталонный Python-конвейер для чтения трёхосевого акселерометра, обнаружения
ударного события и расчёта Maximax SRS методом Smallwood.

## Структура

```text
SRS_pipeline_first_run/
├── main.py                    # Запуск из командной строки
├── pipeline.py                # CSV → компенсация → событие → SRS
├── config.py                  # Профили contest/custom
├── csv_io.py                  # Чтение и проверка CSV
├── accelerometer.py           # Компенсация, модуль, детектор
├── smallwood.py               # Рекурсия и SRS
├── frequency_grid.py          # Загрузка/генерация сетки
├── data_models.py             # Общие структуры данных
├── data/
│   └── frequency_grid.csv     # Точная конкурсная сетка, 94 значения
├── examples/
│   └── contest.json           # Настройки тестового запуска
├── input/
│   └── test_accelerometer_xyz.csv
├── output/                    # Здесь появятся результаты
├── tests/
│   └── test_smoke.py
├── requirements.txt
├── run_example.bat
└── run_example.sh
```

## Требования

- Python 3.11 или новее;
- NumPy;
- SciPy — для дальнейших сравнительных тестов с `lfilter`.

## Установка

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Если PowerShell запрещает запуск `Activate.ps1`, виртуальное окружение можно не
активировать:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py input\test_accelerometer_xyz.csv --profile contest --config-file examples\contest.json
```

### Linux

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

## Тестовый запуск

Windows:

```powershell
python main.py input\test_accelerometer_xyz.csv --profile contest --config-file examples\contest.json
```

или двойным щелчком по `run_example.bat`.

Linux:

```bash
python3 main.py input/test_accelerometer_xyz.csv --profile contest --config-file examples/contest.json
```

## Запуск со своим CSV

Файл должен содержать:

```text
t,ax,ay,az
```

Пример запуска с подтверждённым интервалом покоя 0–20 мс:

```bash
python main.py my_accelerometer.csv \
  --profile contest \
  --rest-start 0.0 \
  --rest-end 0.02 \
  --output output/my_srs.csv \
  --metadata-output output/my_event.json
```

Интервал покоя должен быть выбран по реальным условиям измерения. Алгоритм не
удаляет гравитацию автоматически без такого интервала.

Для расчёта затухания осцилляторов после конца кадра:

```bash
python main.py my_accelerometer.csv \
  --profile contest \
  --rest-start 0.0 \
  --rest-end 0.02 \
  --tail-mode zero_tail \
  --tail-time-constants 5
```

## Выходные файлы

### `output/srs_result.csv`

```text
f_hz,gx_g,gy_g,gz_g,g_total_g
```

- `f_hz` — собственная частота виртуального осциллятора;
- `gx_g`, `gy_g`, `gz_g` — обязательные SRS по трём осям;
- `g_total_g` — дополнительный SRS полного ускорения.

### `output/event_metadata.json`

Содержит:

- реальную частоту дискретизации;
- параметры компенсации;
- индексы и время границ кадра;
- `Amax`;
- `Q` и режим хвоста;
- максимум каждого спектра и соответствующую частоту;
- предупреждения.

## Проверка

```bash
python -m unittest discover -s tests -v
```

## Важное замечание о сетке

Приложенный официальный файл содержит 94 значения от `10.0000009537` до
`2113.48925781` Гц, хотя исходное требование указывает верхнюю границу 2500 Гц.
Код не достраивает конкурсную сетку самостоятельно и выводит предупреждение.
Перед финальной сдачей нужно подтвердить у организаторов, какой источник
является приоритетным.


## Преобразование бинарного файла STM32 в CSV с ускорениями в g

Конвейер уже предполагает, что `ax`, `ay`, `az` в CSV представлены в **g**:
в `config.py` задано `InputConfig.acceleration_unit = "g"`. Алгоритм
Смолвуда менять не нужно. Пересчёт выполняется в `bin_to_csv.py`.

**ВНИМАНИЕ:** формат `.bin` зависит от прошивки STM32, а не от LSM6DS3.
Без бинарника или исходного кода функции записи нельзя доказать, что
записи действительно состоят из 6 байт (`int16 X,Y,Z`) без меток времени.

Сценарий A: STM32 записывает сырые регистры LSM6DS3 (3 × int16, little-endian):

```powershell
python bin_to_csv.py input\sensor.bin --output input\sensor.csv --format raw16 --full-scale-g 16 --sample-rate 6660
```

* `--full-scale-g` — **реально настроенный** диапазон ±2/4/8/16g в `CTRL1_XL`, а не предполагаемое число.
* `--sample-rate` — **реальная** частота регистрации. Если она не 6660 Гц,
  поменяйте `sample_rate_hz` в профиле custom или согласуйте настройки pipeline.
* `--byteorder big` — если порядок байтов отличается от обычного little-endian.
* `--header-bytes N`, `--record-bytes N`, `--xyz-offset N` — для известного
  фиксированного формата STM32 с заголовком или дополнительными полями;
  временные метки из таких записей этот конвертер не извлекает.

Сценарий B: STM32 сохраняет три `float32` уже в g:

```powershell
python bin_to_csv.py input\sensor.bin --output input\sensor.csv --format float32-g --sample-rate 6660
```

Сценарий C: STM32 сохраняет `float32` в м/с²:

```powershell
python bin_to_csv.py input\sensor.bin --output input\sensor.csv --format float32-ms2 --sample-rate 6660
```

CSV будет иметь строго нужный заголовок `t,ax,ay,az`, время в секундах,
все ускорения в g. Затем запускайте:

```powershell
python main.py input\sensor.csv --profile contest --config-file examples\contest.json --rest-start 0.0 --rest-end 0.01
```

Выбирайте интервал покоя по **реальному** измерению, а не автоматически.
Для профиля `contest` ожидается частота около 6660 Гц.
Разметка журнала MCU, диапазон акселерометра, возможный заголовок и реальные
временные метки должны быть подтверждены отдельно.
