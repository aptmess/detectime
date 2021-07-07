# Detectime

Данный репозиторий представляет из себя решение соревнования [MCS 2021. Gesture recognition](https://boosters.pro/championship/machinescansee2021/overview), 
проходившим с 6 июня по 5 июля 2021 года, на котором участникам предстояло решить задачу распознования одного из 6 жестов:

- `stop`
- `victory`
- `mute`
- `ok`
- `like`
- `dislike`

## Algorithm
Решение данной задачи было построено на предположении, что при показывании жеста рука человека находится недалеко от его лица.

На основании данного предположения автором был предложен следующий алгоритм решения данной задачи:

1. Поиск всевозможных лиц на входном изображении с определённым порогом уверенности (`default=0.5`). Применялась модель [RetinaNetMobileNetV1](https://github.com/hukkelas/DSFD-Pytorch-Inference)
2. Поиск всевозможных рук на входном изображении (`default=0.9901`). Применялась модель [MASK RCNN](https://github.com/matterport/Mask_RCNN) и, модифицированная под распознование рук модель [Hand_RCNN](https://github.com/SupreethN/Hand-CNN). 
При работе использовались предобученные [веса](https://github.com/theerapatkitti/hand_mask_rcnn/releases/tag/1.0).
3. Среди пар `лицо-рука` выбирается пара с наибольшей площадью пересечения руки и лица, увеличенного в `crop_coeff=1.5` раза:  
    - если наибольшая площадь пересечения равна нулю, то рука находится далеко от лица, и, соответственно, картинка помечалась меткой `no_gesture`;
    - если же площадь пересечения являлась ненулевой, то входное изображение обрезалось до размеров найденной наиболее вероятной для показа жеста руки;
4. Обученная модель классификации (использовались вариации моделей [ResNet](https://pytorch.org/vision/stable/models.html)) предсказывает вероятность того или иного жеста.

Все вычисления производились в среде `google.colab`.
## Quick Start

- Создайте папку `./model/` в корне репозитория и скачайте веса предобученных моделей в данную папку. Веса скачиваются по [ссылке](https://drive.google.com/drive/folders/1ZlkedG4JWcSNep_iygLagxsRONT7bUg6?usp=sharing) - общий вес менее 500 MB.

- Установите необходимые зависимости

```bash
pip install -r requirements.txt
```

- Запустите файл `script.py`

```bash
python script.py
```

Модель предскажет вероятности трех картинок формата`.jpg`, расположенных в папке `data/`, а также запишет в файл `answers.csv`.

## Reproduction

Для воспроизведения результатов и дальнейшего их улученя были подготовлены два ноутбука:

- [01. get_train_data.ipynb](https://github.com/aptmess/detectime/blob/main/notebooks/01.%20get_train_data.ipynb) - представляет из себя получение обучающей выборки для модели классификации жестов;
- [02. train_resnet.ipynb](https://github.com/aptmess/detectime/blob/main/notebooks/02.%20train_resnet.ipynb) - представляет из себя непосредственно обучение модели классификации

Для запуска данных ноутбуков необходимо загрузить [данные](https://github.com/aptmess/detectime/blob/main/download.txt), представленные на сореновании, с помощью команды ниже — для этого потребуется около 90 GB на диске или виртуальном хранилище.

```bash
sh download_data.sh
```

- Данная команда автоматически загрузит данные в директорию `./INPUT_DATA/TRAIN_DATA/`.
- Скачанные `zip`- архивы будут доступны в директории `./INPUT_DATA/ZIP/`

## Repo Structure

Данный репозиторий состоит из нескольких логических структурных блоков:

### [./detectime/](https://github.com/aptmess/detectime/tree/main/detectime)

Основной модуль, в котором собран весь код - от инициализации моделей, обучению и предсказанию жестов. Ниже приведены краткие описания каждого из модулей:

- [./detectime/augmentation.py](https://github.com/aptmess/detectime/blob/main/detectime/augmentations.py) - функции и классы для аугментации и операций над изображениями (Resize, Crop);
- [./detectime/averagemeter.py](https://github.com/aptmess/detectime/blob/main/detectime/averagemeter.py) - кастомный подсчет метрик;
- [./detectime/dataset.py](https://github.com/aptmess/detectime/blob/main/detectime/dataset.py) - создание тренировочных и валидационных датасетов;
- [./detectime/detectime.py](https://github.com/aptmess/detectime/blob/main/detectime/detectime.py) - основной код - выполняет работу по иницилазации моделей и сохранению результатов работы;
- [./detectime/detection.py](https://github.com/aptmess/detectime/blob/main/detectime/detection.py) - функции, выполняющие поиск жеста на картинке;
- [./detectime/loss_function.py](https://github.com/aptmess/detectime/blob/main/detectime/loss_function.py) - loss функции;
- [./detectime/maskrcnn.py](https://github.com/aptmess/detectime/blob/main/detectime/maskrcnn.py) - модуль для инициализации модели распознования рук на картинке;
- [./detectime/models.py](https://github.com/aptmess/detectime/blob/main/detectime/models.py) - загрузка моделей из `torchvision.models`;
- [./detectime/optimizers.py](https://github.com/aptmess/detectime/blob/main/detectime/optimizers.py) - различные оптимизаторы;
- [./detectime/train.py](https://github.com/aptmess/detectime/blob/main/detectime/train.py) - тренировка и валидация моделей;
- [./detectime/utils.py](https://github.com/aptmess/detectime/blob/main/detectime/utils.py) - различные помогающие функции.

### [./mrcnn/](https://github.com/aptmess/detectime/tree/main/mrcnn)

В данном модуле расположена реализация и иницализация модели [HandRCNN](https://github.com/SupreethN/Hand-CNN), перенесенная на версию `tensorflow 2.5.0`. 
Модель в данном модуле осуществляет поиск рук на картинке.

### `./model/`

В данной директории расположены веса предобученных моделей. В изначальной версии данного репозитория в данной папке находятся следующие веса:

- `gesture_classification.pth` - веса модели `ResNet34`, обученная на 50000 изображениях найденных жестов на картинках с вероятностью 0.99;
- `mask_rcnn_hand_detection.h5` - веса модели `HandRCNN`. Взяты из [данного репозитория](https://github.com/theerapatkitti/hand_mask_rcnn/releases/tag/1.0).

### [./data/](https://github.com/aptmess/detectime/tree/main/data)

Данная директория содержит необходимые данные для обучения.

- `*.jpg` pictures and `test.csv` file - необходимы для кастомизации структуры работы, как в `docker`-контейнере на соревновании;
- [./data/INPUT_DATA/](https://github.com/aptmess/detectime/tree/main/data/INPUT_DATA) - содержит исходные данные соревнования, а именно разметку `train.csv`
 и пример сабмита;
- `./data/INPUT_DATA/JSON` - содержит `*.json` файлы:
    - с найденными для каждого изображения лицами `train_with_bboxes.json`;
    - найденными жестами (руками) `hands.json`;
    - разбиение файла `hands.json` на обучающую и валидационную выборку;
- `.data/INPUT_DATA/TRAIN_DATA` - содержит директории и исходными данными, 
соответствующие путям из столбца исходных данных в `train.csv - frame_path`
- `.data/INPUT_DATA/ZIP` - содержит загруженные данные в `zip`-архивах, опционально
- `.data/experiments` - директория, содержащая веса обучающихся моделей, сохраняются через функцию `detectime.utils.save_checkpoint`

### [./notebooks/](https://github.com/aptmess/detectime/tree/main/notebooks)

Как и было сказано, сожержит подробное описание по воспроизведению результатов и их возможному улучшению.


### Additional

- [./answers.csv](https://github.com/aptmess/detectime/blob/main/answers.csv) - сюда записываются ответы в соревновании;
- [./config.yml](https://github.com/aptmess/detectime/blob/main/config.yml) - основной конфиг с возможностью настройки. Для запуска удобно сделать параметр `config.utils.show_gesture_prediction_result = False`;
- [./definitions.py](https://github.com/aptmess/detectime/blob/main/definitions.py) - содержит абсолютный путь `root` директории;
- [./download.txt](https://github.com/aptmess/detectime/blob/main/download.txt) - ссылки на скачивание данных через `wget -u download.txt`;
- [./download_data.sh](https://github.com/aptmess/detectime/blob/main/download_data.sh) - `bash` скрипт для удобного скачивания всех данных;
- [./log_config.yml](https://github.com/aptmess/detectime/blob/main/log_config.yml) - конфиг для логирования;
- [./requirements.txt](https://github.com/aptmess/detectime/blob/main/requirements.txt) - список зависимостей;
- [./script.py](https://github.com/aptmess/detectime/blob/main/script.py) - основной модуль, запускающий всю систему из `root` директории.

## Links

Неоходимо перечислить еще раз источники, на которые ссылался данный репозиторий:

- [Baseline от организаторов](https://github.com/AlexanderParkin/mcs_gestures_baseline);
- [HandRCNN Tensorflow v1.4.0](https://github.com/SupreethN/Hand-CNN)
- [Weights MaskRCNN](https://github.com/theerapatkitti/hand_mask_rcnn/releases/tag/1.0)

## Improvements

1. Данное решение не смогло пройти 30-минутный барьер на соревновании, соотвественно необходимо думать, как в будущем увеличить скорость;
2. Необходимо переписать в модуле `./mrcnn` часть с кодом, тренирующая модель. Неделями ранее обучить модель у меня не получилось.
3. Каким-то образом узнать все-таки значение метрики на тестовом множестве.
4. Сделать тестовый ноутбук на `google.colab`.


Author
=======

* [![icon][mail]](mailto:improfeo@yandex.ru)
  [![icon][github]](https://github.com/aptmess) 
  [![icon][theme]](https://t.me/aptmess) 
  &nbsp; Aleksandr Shirokov
  
<a href="https://feathericons.com/">Icons by Feather</a>

[mail]: resources/mail.svg
[github]: resources/github.svg
[theme]: resources/airplay.svg
