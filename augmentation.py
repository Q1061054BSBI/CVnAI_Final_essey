import os
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
import random
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import glob
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


def identify_minor_classes(class_counts, threshold=500):
    minor_classes = {cls: count for cls, count in class_counts.items() if count < threshold}
    print("minor_classes:")
    print(minor_classes)
    return minor_classes

def identify_domenant_classes(class_counts, threshold=1500):
   
    domenant_classes = {cls for cls, count in class_counts.items() if count > threshold}
    print("domenant_classes:")
    print(domenant_classes)
    return domenant_classes

def augment_image(image, boxes):
    """
    Применяет аугментацию к изображению и обновляет bounding boxes.
    
    Параметры:
    - image (numpy.ndarray): изображение в формате numpy.
    - boxes (list): список координат bounding boxes.
    
    Возвращает:
    - augmented_image (numpy.ndarray): аугментированное изображение.
    - augmented_boxes (list): обновленные координаты bounding boxes.
    """
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Rotate(limit=15, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=0.5),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=[]))

    transformed = transform(image=image, bboxes=boxes)
    augmented_image = transformed['image'].permute(1, 2, 0).numpy()
    augmented_image = (augmented_image * 255).astype(np.uint8)
    augmented_boxes = transformed['bboxes']
    
    return augmented_image, augmented_boxes

def save_augmented_data(augmented_image, original_annotation_path, augmented_boxes, output_image_path, output_annotation_path):
    """
    Сохраняет аугментированное изображение и обновленную аннотацию в XML.
    
    Параметры:
    - augmented_image (numpy.ndarray): аугментированное изображение.
    - original_annotation_path (str): путь к оригинальной аннотации.
    - augmented_boxes (list): обновленные bounding boxes.
    - output_image_path (str): путь для сохранения изображения.
    - output_annotation_path (str): путь для сохранения XML аннотации.
    """
    Image.fromarray(augmented_image).save(output_image_path)
    tree = ET.parse(original_annotation_path)
    root = tree.getroot()
    
    for obj, box in zip(root.findall('object'), augmented_boxes):
        bndbox = obj.find('bndbox')
        bndbox.find('xmin').text = str(int(box[0]))
        bndbox.find('ymin').text = str(int(box[1]))
        bndbox.find('xmax').text = str(int(box[2]))
        bndbox.find('ymax').text = str(int(box[3]))

    tree.write(output_annotation_path)

def augment_voc_classes(image_names, images_dir, annotations_dir, classes, output_dir, domenant_classes, target_multiplier=2):
    """
    Выполняет аугментацию только для изображений, содержащих объекты меньших классов
    и не содержащих объекты из доминирующих классов.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_images_dir = os.path.join(output_dir, 'Images')
    output_annotations_dir = os.path.join(output_dir, 'Annotations')
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_annotations_dir, exist_ok=True)

    for cls, count in classes.items():
        target_count = count * target_multiplier
        current_count = 0
        
        for image_name in image_names:
            annotation_file = image_name.replace('.jpg', '.xml')
            annotation_path = os.path.join(annotations_dir, annotation_file)
            
            if not os.path.exists(annotation_path):
                continue

            # Загружаем изображение и проверяем наличие только целевого класса
            image, boxes, is_valid = load_image_with_class_check(image_name, images_dir, annotation_path, cls, domenant_classes)
            if not is_valid:
                continue  # Пропускаем изображение, если оно содержит доминирующий класс
            
            for i in range(target_multiplier):
                if current_count >= target_count:
                    break
                
                augmented_image, augmented_boxes = augment_image(image, boxes)
                
                output_image_path = os.path.join(output_images_dir, f"{cls}_aug_{current_count}.jpg")
                output_annotation_path = os.path.join(output_annotations_dir, f"{cls}_aug_{current_count}.xml")
                save_augmented_data(augmented_image, annotation_path, augmented_boxes, output_image_path, output_annotation_path)
                
                current_count += 1

def visualize_random_augmented_image(augmented_images_dir, augmented_annotations_dir):
    """
    Отображает случайное аугментированное изображение с bounding boxes из аннотации.
    
    Параметры:
    - augmented_images_dir (str): директория с аугментированными изображениями.
    - augmented_annotations_dir (str): директория с аннотациями.
    """
    image_files = glob.glob(os.path.join(augmented_images_dir, '*.jpg'))
    random_image_path = random.choice(image_files)
    image_name = os.path.basename(random_image_path)
    annotation_path = os.path.join(augmented_annotations_dir, image_name.replace('.jpg', '.xml'))
    
    if not os.path.exists(annotation_path):
        print(f"No annotation found for image {random_image_path}")
        return
    
    visualize_augmented_image_with_boxes(random_image_path, annotation_path)

def load_image_with_class_check(image_name, images_dir, annotation_path, target_class, dominant_classes):
    """
    Загружает изображение и bounding boxes только для объектов целевого класса, 
    исключая изображения, содержащие объекты преобладающих классов.

    Параметры:
    - image_name (str): имя файла изображения.
    - images_dir (str): директория с изображениями.
    - annotation_path (str): путь к аннотации (XML файл).
    - target_class (str): целевой класс для аугментации (может включать подклассы person).
    - dominant_classes (set): множество преобладающих классов для исключения.

    Возвращает:
    - image (np.array): изображение.
    - boxes (list): список bounding boxes для целевого класса.
    - is_valid (bool): True, если целевой класс найден и преобладающие классы отсутствуют, иначе False.
    """
    image_path = os.path.join(images_dir, image_name)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    boxes = []
    contains_target_class = False
    contains_dominant_class = False
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    for obj in root.findall('object'):
        label = obj.find('name').text
        if label == target_class:
            contains_target_class = True
            bndbox = obj.find('bndbox')
            xmin = int(float(bndbox.find('xmin').text))
            ymin = int(float(bndbox.find('ymin').text))
            xmax = int(float(bndbox.find('xmax').text))
            ymax = int(float(bndbox.find('ymax').text))
            boxes.append([xmin, ymin, xmax, ymax])

        if label in dominant_classes:
            contains_dominant_class = True
            break

    is_valid = contains_target_class and not contains_dominant_class
    return image, boxes, is_valid

def visualize_augmented_image_with_boxes(image_path, annotation_path):
   
    image = Image.open(image_path)
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image)

    tree = ET.parse(annotation_path)
    root = tree.getroot()
    
    for obj in root.findall('object'):
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        
        width = xmax - xmin
        height = ymax - ymin
        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='r', facecolor='none')
        
        ax.add_patch(rect)
        label = obj.find('name').text
        ax.text(xmin, ymin - 10, label, color='red', fontsize=12, weight='bold')

    plt.axis('off')
    plt.show()