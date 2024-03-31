

from image_recognition.modules.importer import importer
from image_recognition.modules.model import model
from image_recognition.modules.visualization import plotter_evaluator, tester
import os

def main():  # pragma: no cover

    ## Формирование классов на основе файловой структуры

    # Внутри указанной директории должны нахдиться папки, имя которых соотвествует классу.
    # В папках -- изображения, соответсвующие классу. Называние изображений не имеет значения.

    # data_directory/
    # ...class_a/
    # ......a_image_1.jpg
    # ......a_image_2.jpg
    # ...class_b/
    # ......b_image_1.jpg
    # ......b_image_2.jpg

    # Размер по вертикали, размер по горизонтали. К этим значениям будут приведены все изображения (сжаты/растянуты, не обрезаны)
    height = 140
    width = 90
    image_size = (height, width)
    # Больше -- быстрее, меньше -- точнее. В теории.
    batch_size = 32
    # Вышеописанная директория
    data_directory = "Data50"
    # Разделение на тренировочные и тестовые данные в долях. Указывается доля тестовых данны (.2 по-умолчанию)
    validation_split = 0.2

    i = importer(image_size, batch_size, data_directory, validation_split)
    
    # Получение имён классов, числа классов.
    class_names = i.class_names
    num_classes = i.num_classes

    ## Обработка данных
    # Вносится рандомизация (ротация, зум, перемещение). Также приводится яркость к понятному нейросети формату (вместо 0-255, 0-1).
    # По-умолчанию (0.2, 0.1, 0.08)
    i.generate_augmentation_layers(0.2, 0.1, 0.08)

    ### Применение слоёв обработки данных
    i.apply_augmentation()

    ## Получение данных для модели
    train_ds = i.train_ds
    val_ds = i.validation_ds

    ## Формирование модели
    # Модель последовательная. Состоит из слоёв, каждый из которых исполняется после предыдущего.
    # В первом слое описывается форма подаваемых данных. Первые два параметра -- размеры изображения (описаны в начале).
    # Третий пораметр: 1 -- ч/б изображение, 2 -- RGB, 3 -- RGBA

    # Последний слой имеет число нейронов, равное количеству классов.

    # Слои convolution 2D
    # Первое число -- количество фильтров, второе -- окно в пискислях (3 на 3, напрмиер), которым алгоритм проходит по изображению.
    # Каждый новый tuple -- новый слой Conv2D. Можно эксперементировать с числами.
    conv_descriptor = (
        (8, (3,3)),
        (16, (3,3)),
        (8, (3,3))
    )
    # Количсетво простых слоёв, сильно повышать нет смысла (по умолчанию 1)
    dense_layer_number = 2

    m = model(image_size, num_classes, dense_layer_number, conv_descriptor)
    m.compile()

    # Обучение нейросети
    # Число проходов по набору данных. Не всегда улучшает результат. Надо смотреть на графики. (50 по умолчанию, при малом наборе данных)
    epochs = 50

    # init_ -- методы, добавляющие функции, применяемые после каждой итерации.
    #
    # learning rate reduction уменьшает скорость обучения, когда сеть застряла (не улучшается точность), 
    # что может помочь преодолеть это состояние (overfitting) (по умолчанию включено)
    # save at epoch делает сохранение сети после каждой итерации (по умолчанию отключено)
    m.init_learning_rate_reduction()
    # m.init_save_at_epoch()

    # Сама тренировка
    m.train(train_ds, epochs, val_ds)

    # Получение истории выполеннеия тренировки и модели
    history = m.history
    model_instance = m.model

    # Визуализация
    pe = plotter_evaluator(history, model_instance, class_names)
    pe.calc_predictions(val_ds)

    ## Графики потерь и точности
    # Высокой должна быть и accuracy и val_accuracy. Первая -- точность на обучающей выборке, вторая -- на тестовой. 
    # Когда/если точность на обучающей выборке начинает превосходить точность на тестовой, продолжать обучение не следует.
    # Потери (loss) должны быть низкими.
    pe.plot_loss_accuracy()

    ## Вычисление отчёта о качестве классификации
    # Значения accuracy, recall, f1 должны быть высокими.
    pe.print_report()

    ## Матрица запутанности
    # Хорший способ понять, как именно нейросеть ошибается
    pe.plot_confusion_matrix()


    # Тестирование избранных случаев
    t = tester(model_instance, class_names)

    images = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "zero"]
    # Составной путь. Строится от корня программы
    for i in range(len(images)):

        absolute_path = os.path.join(os.getcwd(), 'test_images', images[i] + '.jpg')
        t.read_image(absolute_path, False)

        t.img_predict(images[i])

    # Импорт тестевого набора данных (существенно больше тренировачного)
    # и последующа обработка plotter_evaluatorом
    import_test = importer(image_size, batch_size, 'Data2k')
    test_ds = import_test.train_ds
    pe.calc_predictions(test_ds)
    pe.print_report()
    pe.plot_confusion_matrix()
