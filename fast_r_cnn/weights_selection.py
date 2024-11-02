from torch.utils.data import WeightedRandomSampler
from collections import Counter
import xml.etree.ElementTree as ET
import os
from data_processing import count_classes, get_person_label

def create_weighted_sampler(train_images, annotations_dir):
    """
    Создает WeightedRandomSampler для тренировочного набора, основываясь на весах классов,
    включая разделение по pose и actions для класса "person".
    
    Параметры:
    - train_images (list): Список файлов изображений в тренировочном наборе.
    - annotations_dir (str): Директория с аннотациями изображений.

    Возвращает:
    - WeightedRandomSampler: Самплер для взвешенной выборки изображений.
    """
    # Подсчитываем количество изображений каждого класса
    train_class_counts = count_classes(annotations_dir, train_images)
    total_count = sum(train_class_counts.values())
    class_weights = {cls: total_count / count for cls, count in train_class_counts.items()}

    # Определяем вес для каждого изображения в train_images
    image_weights = []
    for image_file in train_images:
        annotation_file = image_file.replace('.jpg', '.xml')
        annotation_path = os.path.join(annotations_dir, annotation_file)
        
        # Если аннотация отсутствует, пропускаем
        if not os.path.exists(annotation_path):
            image_weights.append(1)
            continue
        
        # Определяем классы на изображении и рассчитываем вес изображения
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        
        total_image_weight = 0
        object_count = 0
        for obj in root.findall('object'):
            label = get_person_label(obj)
            # label = obj.find('name').text
            # # Логика для класса "person" с учетом pose и actions
            # if label == "person":
            #     pose = obj.find('pose').text if obj.find('pose') is not None else 'Unspecified'
                
            #     if pose != "Unspecified":
            #         label = f"person_{pose.lower()}"
            #     else:
            #         actions = obj.find('actions')
            #         action_found = False
            #         if actions is not None:
            #             for action in actions:
            #                 if int(action.text) == 1:
            #                     label = f"person_{action.tag.lower()}"
            #                     action_found = True
            #                     break
            #         if not action_found:
            #             label = "person_unspecified"

            total_image_weight += class_weights.get(label, 1)
            object_count += 1
        
        image_weight = total_image_weight / object_count if object_count > 0 else 1
        image_weights.append(image_weight)

    # Создаем WeightedRandomSampler
    sampler = WeightedRandomSampler(image_weights, num_samples=len(image_weights), replacement=True)
    return sampler