import os
import xml.etree.ElementTree as ET
import shutil
from collections import defaultdict

# Оригинальные директории
original_images_dir = './datasets/Images'
original_annotations_dir = './datasets/Annotations'

# Директории для подмножества данных
small_train_images_dir = './datasets/small/train/Images'
small_train_annotations_dir = './datasets/small/train/Annotations'
os.makedirs(small_train_images_dir, exist_ok=True)
os.makedirs(small_train_annotations_dir, exist_ok=True)

all_classes = []
# Список всех классов
# all_classes = [
#     'person', 'car', 'dog', 'cat', 'bicycle', 'truck', 'bird', 'bus', 'horse', 'sheep',
#     'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
#     'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports_ball', 'kite',
#     'baseball_bat', 'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket',
#     'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon'
# ]

# Словарь для отслеживания выбранных изображений для каждого класса
class_images = defaultdict(list)

# Проходим по всем аннотациям и группируем изображения по классам
for annotation_file in os.listdir(original_annotations_dir):
    if annotation_file.endswith('.xml'):
        annotation_path = os.path.join(original_annotations_dir, annotation_file)
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        
        image_name = annotation_file.replace('.xml', '.jpg')
        image_path = os.path.join(original_images_dir, image_name)
        
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name not in all_classes:
                all_classes.append(class_name)
            class_images[class_name].append((image_name, annotation_file))

print(all_classes)
print(len(class_images))
# Создаем подмножество, чтобы включить хотя бы одно изображение для каждого класса
selected_images = set()
counter = 0;
while len(selected_images)<100:
    for class_name, images in class_images.items():
        if images:
            # Берем одно изображение для каждого класса
            selected_image, selected_annotation = images[counter]
            selected_images.add((selected_image, selected_annotation))
    counter += 1

# Копируем выбранные файлы в новую директорию
for image_name, annotation_file in selected_images:
    shutil.copy(os.path.join(original_images_dir, image_name), small_train_images_dir)
    shutil.copy(os.path.join(original_annotations_dir, annotation_file), small_train_annotations_dir)

print("Подмножество данных с объектами всех классов создано.")