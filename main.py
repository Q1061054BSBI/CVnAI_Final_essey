import os
import random
import shutil
import torch
from ultralytics import YOLO
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import precision_score, recall_score, average_precision_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from custom_dataset import CustomDataset


import ssl
import certifi
import warnings

warnings.filterwarnings("ignore")
ssl._create_default_https_context = ssl._create_unverified_context 
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())


dataset_dir = './datasets/Images'
annotations_dir = './dataset/Annotations'

train_dir = './datasets/small/train'
val_dir = './datasets/val'
test_dir = './datasets/test'

# Создание папок, если их нет
for dir in [train_dir, val_dir, test_dir]:
    os.makedirs(os.path.join(dir, 'Images'), exist_ok=True)
    os.makedirs(os.path.join(dir, 'Annotations'), exist_ok=True)

# Получение всех файлов изображений
# all_images = [f for f in os.listdir(dataset_dir) if f.endswith('.jpg')]

# Разделение на train, val и test
# train_images, temp_images = train_test_split(all_images, test_size=0.3, random_state=42)
# val_images, test_images = train_test_split(temp_images, test_size=0.5, random_state=42)

# Функция для копирования файлов
# def copy_files(image_list, target_dir):
#     for img in image_list:
#         img_path = os.path.join(dataset_dir, img)
#         annot_path = os.path.join(annotations_dir, img.replace('.jpg', '.xml'))

#         # Копирование изображений
#         shutil.copy(img_path, os.path.join(target_dir, 'Images', img))

#         # Копирование аннотаций
#         if os.path.exists(annot_path):
#             shutil.copy(annot_path, os.path.join(target_dir, 'Annotations', img.replace('.jpg', '.xml')))

# Копирование файлов в соответствующие папки
# copy_files(train_images, train_dir)
# copy_files(val_images, val_dir)
# copy_files(test_images, test_dir)

# Параметры обучения
num_epochs = 5
batch_size = 16
learning_rate = 0.001

# Подготовка данных

# Трансформации для изображений
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor()
])

# Создание датасетов и загрузчиков
train_dataset = CustomDataset(images_dir=f"{train_dir}/Images", annotations_dir=f"{train_dir}/Annotations", transform=transform)
if len(train_dataset) == 0:
    raise ValueError("train_dataset пустой, проверьте данные.")

val_dataset = CustomDataset(images_dir=f"{val_dir}/Images", annotations_dir=f"{val_dir}/Annotations", transform=transform)
if len(val_dataset) == 0:
    raise ValueError("val_dataset пустой, проверьте данные.")

test_dataset = CustomDataset(images_dir=f"{test_dir}/Images", annotations_dir=f"{test_dir}/Annotations", transform=transform)
if len(test_dataset) == 0:
    raise ValueError("test_dataset пустой, проверьте данные.")

def custom_collate_fn(batch):
    batch = [b for b in batch if b is not None]  # Удаляем пустые элементы
    return tuple(zip(*batch)) if batch else ([], [])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

for img, target in train_loader:
    if len(target) == 0 or all(len(t['boxes']) == 0 for t in target):
        print("Empty batch found")
    else:
        print("Batch contains data")

val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

# Загрузка модели YOLO
model = YOLO('yolov5s.pt')  # Вы можете заменить 'yolov5s.pt' на другую версию модели (yolov5m.pt, yolov5l.pt и т.д.)

# Настройка обучения
model.train(
    data='dataset_config.yaml',
    epochs=num_epochs,
    imgsz=640,
    lr0=learning_rate
)

# Оценка модели на тестовом наборе
def evaluate_model(model, data_loader, iou_threshold=0.5):
    model.eval()
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for images, targets in data_loader:
            predictions = model(images)
            for i, prediction in enumerate(predictions):
                all_targets.append(targets[i])
                all_predictions.append(prediction)

    # Расчет метрик (при необходимости, добавить собственные метрики)
    precision = precision_score(all_targets, all_predictions, average='macro')
    recall = recall_score(all_targets, all_predictions, average='macro')
    ap = average_precision_score(all_targets, all_predictions, average='macro')

    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, AP: {ap:.4f}")

# Вызов оценки
evaluate_model(model, test_loader)
