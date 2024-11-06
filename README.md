# YandexCup 2024: Music Information Retrieval

## Установка окружения

1. Создайте и активируйте окружение с Python 3.12:

    ```bash
    conda create -n yacup python=3.12
    conda activate yacup
    ```

2. Установите Poetry для управления зависимостями и настройте проект:

    ```bash
    pip install poetry
    poetry install
    ```

3. Установите дополнительные зависимости для PyTorch с поддержкой CUDA:

    ```bash
    poetry run poe torch-cuda
    ```

## Подготовка данных

1. Скачайте чекпоинты и данные по ссылкам, и поместите их в корень репозитория:

    - [Чекпоинты](https://disk.yandex.ru/d/l2WDOIOUNAkHFw)
    - [Данные](https://disk.yandex.ru/d/1ULcrmDN273rkA)

2. Распакуйте скачанные архивы:

 ```bash
 unzip artifacts_hgnetv2_b5_metric_learning_big_margin_drop_cliques_test_0_6_6folds02_19_23.zip
 unzip data.zip
 ```

## Запуск

Запустите скрипт для формирования сабмишена из соло-модели:

```bash
poetry run python -m submission
```

## Описание решения

1. Скопировал исходный сигнал в 3 канала и использовал предобученные CNN-бэкбоны без фриза весов. Наилучший результат показал `hgnetv2_b5`, обученный на кросс-энтропию.
2. Применил аугментации: MixUp, CutMix и CutMix по времени.
3. Очистил тренировочный набор от треков с почти константными значениями (`notebooks/Filter.ipynb`)
4. На валидации выбросил клики, внутри которых модель считала все треки далекими друг от друга. Что-то вроде выбросов, которые сложно моделировать (`notebooks/ValScoring.ipynb`).
5. В тесте выбрал пары треков, которые модель считала близкими - обозначил их за ребра и сформировал граф. Посчитал связные компоненты и добавил их как новые клики в трейн (`notebooks/TestScoring.ipynb`)
6. Обучил модель на новом трейне: сначала на кросс-энтропию, затем дообучил с замороженными слоями, используя metric learning. При дообучении использовал другие аугментации - гауссовкий блюр и шум
7. Финальный результат — бленд моделей, обученных с разными параметрами и функциями потерь (`notebooks/MLFLOW.ipynb`)
