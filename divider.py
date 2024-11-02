import os
import shutil
import random
import xml.etree.ElementTree as ET
from processor import convert_voc_to_yolo, create_class_dict, get_person_label
from resizer import resize_image_and_annotations
import cv2


os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'

cv2.setLogLevel(3) 

def is_image_valid(image_path):
    """
    Проверяет, можно ли корректно открыть изображение без ошибок.
    """
    image = cv2.imread(image_path)
    return image is not None

def split_dataset(base_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Разделяет исходный датасет Pascal VOC на тренировочный, валидационный и тестовый наборы.
    Создает структуру папок, требуемую для YOLO и использует функцию для конвертации аннотаций.
    
    Параметры:
    - base_dir (str): Путь к папке с исходными данными (JPEGImages и Annotations).
    - output_dir (str): Путь к папке для сохранения разделенного датасета.
    - train_ratio (float): Доля тренировочного набора.
    - val_ratio (float): Доля валидационного набора.
    - test_ratio (float): Доля тестового набора.
    """
    images_dir = os.path.join(base_dir, 'JPEGImages')
    annotations_dir = os.path.join(base_dir, 'Annotations')

    class_dictionary = create_class_dict(annotations_dir)
    print("\nDictionary:")
    print(class_dictionary)

    # Проверка существования папок
    if not os.path.exists(images_dir) or not os.path.exists(annotations_dir):
        raise FileNotFoundError("Папки JPEGImages и/или Annotations не найдены.")

    # Получение списка всех изображений
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    random.shuffle(image_files)

    # Определение количества файлов для каждого набора
    total_images = len(image_files)
    train_count = int(total_images * train_ratio)
    val_count = int(total_images * val_ratio)

    train_files = image_files[:train_count]
    val_files = image_files[train_count:train_count + val_count]
    test_files = image_files[train_count + val_count:]

    # Создание папок для YOLO
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)

    # Копирование файлов и конвертация аннотаций
    def copy_and_convert_files(file_list, split):
        for file_name in file_list:
            image_src = os.path.join(images_dir, file_name)
            if not is_image_valid(image_src):
                print(f"Пропущено поврежденное изображение: {file_name}")
                continue

            annotation_src = os.path.join(annotations_dir, file_name.replace('.jpg', '.xml'))

            image_dest = os.path.join(output_dir, split, 'images', file_name)
            label_dest = os.path.join(output_dir, split, 'labels', file_name.replace('.jpg', '.txt'))

            shutil.copy(image_src, image_dest)

            if os.path.exists(annotation_src):
                convert_voc_to_yolo(annotation_src, label_dest, image_src, class_dictionary)

    copy_and_convert_files(train_files, 'train')
    copy_and_convert_files(val_files, 'val')
    copy_and_convert_files(test_files, 'test')

    # only for tests
    create_small_train_set(test_files, images_dir, annotations_dir, output_dir, class_dictionary)

    print("Датасет успешно разделен на train, val и test наборы и аннотации преобразованы в YOLO формат.")

def create_small_train_set(test_files, images_dir, annotations_dir, output_dir, class_dictionary, sample_size=100):
    """
    Создает маленький тренировочный набор из 100 изображений с аннотациями для всех классов.

    Параметры:
    - test_files (list): Список файлов из тестового набора.
    - images_dir (str): Путь к папке с изображениями.
    - annotations_dir (str): Путь к папке с аннотациями XML.
    - output_dir (str): Путь к папке для сохранения маленького набора.
    - class_dictionary (dict): Словарь классов с ID.
    - sample_size (int): Количество изображений в маленьком наборе.
    """
    small_train_dir = os.path.join(output_dir, 'small_train')
    os.makedirs(os.path.join(small_train_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(small_train_dir, 'labels'), exist_ok=True)

    selected_images = []
    used_classes = set()

    for file_name in test_files:
        if len(selected_images) >= sample_size:
            break

        annotation_path = os.path.join(annotations_dir, file_name.replace('.jpg', '.xml'))
        if os.path.exists(annotation_path):
            tree = ET.parse(annotation_path)
            root = tree.getroot()

            image_classes = {get_person_label(obj) for obj in root.findall('object')}
            if image_classes.isdisjoint(used_classes) or len(used_classes) < len(class_dictionary):
                selected_images.append(file_name)
                used_classes.update(image_classes)

                image_src = os.path.join(images_dir, file_name)
                image_dest = os.path.join(small_train_dir, 'images', file_name)
                shutil.copy(image_src, image_dest)

                label_dest = os.path.join(small_train_dir, 'labels', file_name.replace('.jpg', '.txt'))
                convert_voc_to_yolo(annotation_path, label_dest, image_src, class_dictionary)

    print(f"Создан маленький тренировочный набор из {len(selected_images)} изображений.")



