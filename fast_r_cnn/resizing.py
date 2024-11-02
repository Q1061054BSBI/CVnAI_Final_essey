import albumentations as A
import cv2
import xml.etree.ElementTree as ET
import os


def resize_images_and_boxes(image_names, images_dir, annotations_dir, output_images_dir, output_annotations_dir, target_size=(224, 224)):
    """
    Изменяет размер изображений и обновляет bounding boxes для массива изображений и их аннотаций.
    
    Параметры:
    - image_names (list): список имен файлов изображений.
    - images_dir (str): директория с исходными изображениями.
    - annotations_dir (str): директория с исходными аннотациями.
    - output_images_dir (str): директория для сохранения изменённых изображений.
    - output_annotations_dir (str): директория для сохранения обновлённых аннотаций.
    - target_size (tuple): конечный размер изображений (ширина, высота).
    """
    # Создаём выходные директории, если они не существуют
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_annotations_dir, exist_ok=True)
    
    # Настройка аугментации для изменения размера
    transform = A.Compose([
        A.Resize(width=target_size[0], height=target_size[1])
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=[]))

    for image_name in image_names:
        image_path = os.path.join(images_dir, image_name)
        annotation_path = os.path.join(annotations_dir, image_name.replace('.jpg', '.xml'))

        if not os.path.exists(annotation_path):
            print(f"Annotation not found for {image_name}, skipping.")
            continue

        # Загружаем изображение и аннотацию
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = load_boxes_from_annotation(annotation_path)

        # Применяем изменение размера
        transformed = transform(image=image, bboxes=boxes)
        resized_image = transformed['image']
        resized_boxes = transformed['bboxes']

        # Сохраняем изменённое изображение
        output_image_path = os.path.join(output_images_dir, image_name)
        cv2.imwrite(output_image_path, cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR))

        # Сохраняем обновленную аннотацию
        output_annotation_path = os.path.join(output_annotations_dir, image_name.replace('.jpg', '.xml'))
        save_resized_annotation(annotation_path, resized_boxes, output_annotation_path, target_size)

def load_boxes_from_annotation(annotation_path):
    """
    Загружает bounding boxes из XML аннотации.
    
    Параметры:
    - annotation_path (str): путь к XML файлу аннотации.
    
    Возвращает:
    - boxes (list): список bounding boxes в формате [xmin, ymin, xmax, ymax].
    """
    boxes = []
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    for obj in root.findall('object'):
        bndbox = obj.find('bndbox')
        xmin = int(float(bndbox.find('xmin').text))
        ymin = int(float(bndbox.find('ymin').text))
        xmax = int(float(bndbox.find('xmax').text))
        ymax = int(float(bndbox.find('ymax').text))
        boxes.append([xmin, ymin, xmax, ymax])

    return boxes

def save_resized_annotation(original_annotation_path, resized_boxes, output_annotation_path, target_size):
    """
    Сохраняет обновленную аннотацию XML после изменения размеров изображения.
    
    Параметры:
    - original_annotation_path (str): путь к оригинальной аннотации.
    - resized_boxes (list): список обновленных bounding boxes.
    - output_annotation_path (str): путь для сохранения новой аннотации.
    - target_size (tuple): конечный размер изображения.
    """
    tree = ET.parse(original_annotation_path)
    root = tree.getroot()
    
    # Обновляем размер изображения в аннотации
    size_element = root.find('size')
    size_element.find('width').text = str(target_size[0])
    size_element.find('height').text = str(target_size[1])
    
    # Обновляем bounding boxes
    object_elements = root.findall('object')
    for obj, box in zip(object_elements, resized_boxes):
        bndbox = obj.find('bndbox')
        bndbox.find('xmin').text = str(int(box[0]))
        bndbox.find('ymin').text = str(int(box[1]))
        bndbox.find('xmax').text = str(int(box[2]))
        bndbox.find('ymax').text = str(int(box[3]))
    
    # Сохраняем обновленную аннотацию
    tree.write(output_annotation_path)