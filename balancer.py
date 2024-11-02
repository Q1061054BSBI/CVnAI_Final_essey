import os
import shutil
import cv2
import numpy as np
import albumentations as A
from collections import Counter
from ploter import count_classes, plot_class_distribution, visualize_random_augmented_image
from processor import create_class_dict

os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'

cv2.setLogLevel(3) 

def augment_image(image_path, bboxes):
    """
    Применяет аугментацию к изображению и обновляет bounding boxes.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Ошибка при чтении изображения: {image_path}")

    try:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        raise ValueError(f"Ошибка при преобразовании изображения в RGB: {image_path}, {e}")


    
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5)
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['labels']))

    labels = [0] * len(bboxes)  # Заглушка для labels, так как они уже есть в txt файле
    
    try:
        transformed = transform(image=image, bboxes=bboxes, labels=labels)
    except Exception as e:
        raise ValueError(f"Ошибка при аугментации изображения: {image_path}, {e}")


    return transformed['image'], transformed['bboxes']

def augment_minor_classes(images_dir, labels_dir, class_dict, lower_threshold, upper_threshold, output_dir):
    """
    Аугментация изображений с малым количеством объектов класса.
    """
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)

    # Получение количества объектов каждого класса
    class_counts = count_classes(labels_dir, class_dict)

    # Определение малых и преобладающих классов
    minor_classes = {cls: count for cls, count in class_counts.items() if count < lower_threshold}
    dominant_classes = {cls for cls, count in class_counts.items() if count > upper_threshold}

    print(f"Minor classes: {minor_classes}")
    print(f"Dominant classes: {dominant_classes}")

    # Поиск изображений для аугментации
    for label_file in os.listdir(labels_dir):
        label_path = os.path.join(labels_dir, label_file)
        image_name = label_file.replace('.txt', '.jpg')
        image_path = os.path.join(images_dir, image_name)

        with open(label_path, 'r') as file:
            lines = file.readlines()
            image_bboxes = []
            contains_minor_class = False
            classes_in_image = set()

            for line in lines:
                parts = line.strip().split()
                class_id = int(parts[0])
                class_name = [k for k, v in class_dict.items() if v == class_id][0]

                # Проверяем, содержит ли изображение малый класс и исключаем преобладающие классы
                if class_name in minor_classes:
                    contains_minor_class = True
                classes_in_image.add(class_name)

                # Получаем координаты bbox
                x_center, y_center, width, height = map(float, parts[1:])
                image_bboxes.append([x_center, y_center, width, height])

            if contains_minor_class and not any(cls in dominant_classes for cls in classes_in_image):
                for i in range(2):  # Создаем 200% увеличение
                    augmented_image, augmented_bboxes = augment_image(image_path, image_bboxes)

                    # Сохраняем новое изображение
                    augmented_image_path = os.path.join(output_dir, 'images', f"aug_{i}_{image_name}")
                    cv2.imwrite(augmented_image_path, cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR))

                    # Сохраняем новую аннотацию
                    augmented_label_path = os.path.join(output_dir, 'labels', f"aug_{i}_{label_file}")
                    with open(augmented_label_path, 'w') as out_file:
                        for bbox in augmented_bboxes:
                            out_file.write(f"{class_id} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")

    print("Аугментация завершена.")


# # Параметры
# # images_dir = './datasets/resized/train/images'
# # labels_dir = './datasets/resized/train/labels'
# output_dir = './datasets/resized/train'
# # lower_threshold = 500  # Порог для малых классов
# # upper_threshold = 1500  # Порог для преобладающих классов

# # # Создаем словарь классов
# annotations_dir = './datasets/Annotations'
# class_dict = create_class_dict(annotations_dir)

# # # Вызов функции дл
# # augment_minor_classes(images_dir, labels_dir, class_dict, lower_threshold, upper_threshold, output_dir)

# classes_val = count_classes(f"{output_dir}/labels", class_dict=class_dict)
# #temporary for tests
# plot_class_distribution(classes_val, "Augumanted images");

# # for i in range(10):
# #     visualize_random_augmented_image(f"{output_dir}/images", f"{output_dir}/labels", class_dict=class_dict)


