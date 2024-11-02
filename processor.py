import os
from PIL import Image
import xml.etree.ElementTree as ET
import cv2
os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'

cv2.setLogLevel(3) 

def convert_voc_to_yolo(annotation_path, output_path, image_path, dictionary):
    """
    Конвертирует аннотацию Pascal VOC в формат YOLO для одного изображения.

    Параметры:
    - annotation_path (str): Путь к XML файлу аннотации.
    - output_path (str): Путь для сохранения .txt файла аннотации.
    - image_path (str): Путь к изображению для определения размеров.
    - dictionary (dict): Словарь классов для преобразования имени класса в ID.
    """
    if not annotation_path.endswith('.xml'):
        raise ValueError("Файл аннотации должен быть в формате XML.")

    if not os.path.exists(annotation_path) or not os.path.exists(image_path):
        print(f"Файл аннотации {annotation_path} или изображение {image_path} не найдено.")
        return

    tree = ET.parse(annotation_path)
    root = tree.getroot()

    image = Image.open(image_path)
    image_width, image_height = image.size

    with open(output_path, 'w') as label_file:
        for obj in root.findall('object'):
            class_name = get_person_label(obj)
            class_id = dictionary.get(class_name, -1)  # Получаем ID класса из словаря
            if class_id == -1:
                print(f"Класс {class_name} не найден в словаре. Пропуск.")
                continue

            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)

            # Преобразование координат в YOLO формат
            x_center = ((xmin + xmax) / 2) / image_width
            y_center = ((ymin + ymax) / 2) / image_height
            box_width = (xmax - xmin) / image_width
            box_height = (ymax - ymin) / image_height

            label_file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")

    # print(f"Конвертация аннотации для {annotation_path} завершена.")



def create_class_dict(annotations_dir):
    unique_classes = set()

    # Обходим все файлы аннотаций и извлекаем имена классов
    for annotation_file in os.listdir(annotations_dir):
        if annotation_file.endswith('.xml'):
            annotation_path = os.path.join(annotations_dir, annotation_file)
            tree = ET.parse(annotation_path)
            root = tree.getroot()

            for obj in root.findall('object'):
                class_name = get_person_label(obj)
                unique_classes.add(class_name)

    # Создаем словарь с ID для каждого класса
    class_to_id = {cls: idx for idx, cls in enumerate(sorted(unique_classes))}
    return class_to_id

def class_name_to_id(class_name, disctionary):
        return disctionary.get(class_name, -1)  # Возвращает -1, если класс не найден

def get_person_label(obj):
    #dividing person into several clases
    label = obj.find('name').text

    # Проверяем, что это действительно "person"
    if label != "person":
        return label  # Возвращаем исходную метку для других классов

    # Проверяем наличие action
    actions = obj.find('actions')
    action_found = False
    if actions is not None:
        all_zero = True  # Флаг, указывающий, что все действия равны нулю
        for action in actions:
            if int(action.text) == 1:
                label = f"person_{action.tag.lower()}"
                action_found = True
                break
            if int(action.text) != 0:
                all_zero = False
        
        # Если все действия равны нулю, отмечаем как unspecified
        if not action_found and all_zero:
            label = "person_unspecified"
    
    # Если action не найден или он unspecified, проверяем pose
    if not action_found or label == "person_unspecified":
        pose = obj.find('pose').text if obj.find('pose') is not None else 'Unspecified'
        if pose != "Unspecified":
            label = f"person_{pose.lower()}"
        else:
            label = "person_unspecified"

    return label

