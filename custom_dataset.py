import os
import torch
from PIL import Image
import xml.etree.ElementTree as ET
import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2

# Создайте аугментации
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Rotate(limit=15, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=0.5),
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=[]))

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, annotations_dir, transform):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
        self.unique_classes = self.get_unique_classes(annotations_dir, self.image_files)
        self.class_to_id = {cls: idx for idx, cls in enumerate(self.unique_classes)}  # Создаем словарь классов

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        annotation_path = os.path.join(self.annotations_dir, img_name.replace('.jpg', '.xml'))
        
        # Загружаем изображение
        img = Image.open(img_path).convert("RGB")
        # img = np.array(img)  # Преобразуем в массив NumPy

        boxes, labels = [], []
        if os.path.exists(annotation_path):
            tree = ET.parse(annotation_path)
            root = tree.getroot()
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                class_id = self.class_name_to_id(class_name)
                if class_id == -1:
                    continue  # Пропускаем, если класс не найден в словаре
                labels.append(class_id)
                bndbox = obj.find('bndbox')
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)
                if xmax > xmin and ymax > ymin:  # Проверка, что боксы имеют положительные размеры
                    boxes.append([xmin, ymin, xmax, ymax])

        # Проверяем, что boxes и labels не пустые
        if len(boxes) == 0 or len(labels) == 0:
            print(f"Skipping image {img_name} due to empty boxes or labels.")
            return self.__getitem__((idx + 1) % len(self.image_files))  # Переходим к следующему изображению

        if self.transform:
            img = self.transform(img)
        
        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64)
        }
        return img, target

    def class_name_to_id(self, class_name):
        return self.class_to_id.get(class_name, -1)  # Возвращаем -1, если класс не найден

    @staticmethod
    def get_unique_classes(annotations_dir, image_files):
        unique_classes = set()
        for image_file in image_files:
            annotation_file = image_file.replace('.jpg', '.xml')
            annotation_path = os.path.join(annotations_dir, annotation_file)
            if os.path.exists(annotation_path):
                tree = ET.parse(annotation_path)
                root = tree.getroot()
                for obj in root.findall('object'):
                    label = obj.find('name').text
                    unique_classes.add(label)
        return list(unique_classes)