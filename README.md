# YandexCup 2024: Music Information Retrieval

Этот проект создан для участия в YandexCup 2024 в категории Music Information Retrieval (MIR). Здесь собраны команды для настройки окружения, установки зависимостей и запуска пайплайна.

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

1. Скачайте чекпоинты и данные по ссылкам, предоставленным организаторами, и поместите их в корень репозитория:

    - [Чекпоинты](https://disk.yandex.ru/d/l2WDOIOUNAkHFw)
    - [Данные](https://disk.yandex.ru/d/1ULcrmDN273rkA)

2. Распакуйте скачанные архивы:

    ```bash
    unzip artifacts_hgnetv2_b5_metric_learning_big_margin_drop_cliques_test_0_6_6folds02_19_23.zip
    unzip data.zip
    ```

## Запуск

Запустите скрипт для формирования сабмишена:

```bash
poetry run python -m submission
