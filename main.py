# main.py
import os
import random
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from voc_dataset import VOCDataset  # Импортируем класс VOCDataset
from data_processing import (visualize_image_and_tensor, validate_labels, 
                             count_classes, plot_class_distribution, 
                             filter_person_and_other_images, count_poses, 
                             plot_pose_distribution)
from torchvision.transforms import functional as F
from augmentation import augment_image, identify_minor_classes, identify_domenant_classes, visualize_augmented_image_with_boxes, visualize_random_augmented_image, augment_voc_classes
from resizing import resize_images_and_boxes
from weights_selection import create_weighted_sampler

import ssl
import certifi

ssl._create_default_https_context = ssl._create_unverified_context 
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())


# start links
data_dir = './dataset/Images'
annotations_dir = './dataset/Annotations'
train_split = 0.7
val_split = 0.15
test_split = 0.15
num_epochs = 5

# fix random seed for repeating in rnadomizer
random.seed(42)

# 
images = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]
#divide all not annotated images
valid_images, black_list = validate_labels(images, annotations_dir, data_dir)

print("Black list (no labels):", len(black_list))

resized_images_dir = './resized_dataset/Images'
resized_annotations_dir = './resized_dataset/Annotations'

resize_images_and_boxes(
    image_names=valid_images,
    images_dir=data_dir,
    annotations_dir=annotations_dir,
    output_images_dir=resized_images_dir,
    output_annotations_dir=resized_annotations_dir,
    target_size=(224, 224)
)

for index in range(10):
    visualize_random_augmented_image(resized_images_dir, resized_annotations_dir)

#after resising the resized dirs become new defaut dirs
data_dir = resized_images_dir
annotations_dir = resized_annotations_dir

random.shuffle(valid_images)


# Разделяем данные
train_size = int(train_split * len(valid_images))
val_size = int(val_split * len(valid_images))
train_images = valid_images[:train_size]
val_images = valid_images[train_size:train_size + val_size]
test_images = valid_images[train_size + val_size:]


# person class dividion distribution
pose_counts = count_poses(annotations_dir, train_images)
plot_pose_distribution(pose_counts, title="Pose and actions Distribution in Training Dataset")

train_class_counts = count_classes(annotations_dir, train_images)
plot_class_distribution(train_class_counts, title="Class Distribution in Training Set")

# validation_class_counts = count_classes(annotations_dir, val_images)
# plot_class_distribution(validation_class_counts, title="Class Distribution in Validation Set")

#detecting classed from train dataset wich need to be augmented
minor_classes = identify_minor_classes(train_class_counts, threshold=500)
domenant_classes = identify_domenant_classes(train_class_counts, threshold=1500)
augment_voc_classes(classes=minor_classes, image_names=train_images, images_dir=data_dir,  annotations_dir=annotations_dir, output_dir='./resized_dataset', target_multiplier=2, domenant_classes=domenant_classes)

# Обновляем train_images, чтобы включить новые аугментированные изображения
augmented_images = [f for f in os.listdir(data_dir) if '_aug_' in f and f.endswith('.jpg')]
train_images.extend(augmented_images)  # Добавляем аугментированные изображения в список тренировочных данных

#check distribution one morw time
train_class_counts = count_classes(annotations_dir, train_images)
plot_class_distribution(train_class_counts, title="Class Distribution in Training Set after augmentation")

# visalisation
for index in range(10):
    visualize_random_augmented_image(data_dir, annotations_dir)



sampler = create_weighted_sampler(train_images, annotations_dir)

#checked till here


# Создаем DataLoader для каждой выборки
train_dataset = VOCDataset(train_images, annotations_dir, data_dir, transforms=F.to_tensor)
val_dataset = VOCDataset(val_images, annotations_dir, data_dir, transforms=F.to_tensor)
test_dataset = VOCDataset(test_images, annotations_dir, data_dir, transforms=F.to_tensor)

train_loader = DataLoader(train_dataset, batch_size=2, sampler=sampler, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

unique_classes = VOCDataset.get_unique_classes(annotations_dir, valid_images)
num_classes = len(unique_classes) + 1  # +1 для фона

print("Unique classes:", unique_classes)
print("Number of classes (including background):", num_classes)

# Инициализируем модель Faster R-CNN с предобученной ResNet50
model = fasterrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torch.nn.Linear(in_features, num_classes)

# Установка устройства
device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
model.to(device)

# Оптимизатор
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

# Функция для оценки на валидационном наборе
def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():  # Отключаем градиенты для оценки
        for images, targets in data_loader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()
    print(f"Validation Loss: {total_loss / len(data_loader)}")
    model.train()  # Возвращаем модель в режим обучения

# Функция для обучения одной эпохи
def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    total_loss = 0
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()
    print(f"Epoch {epoch}, Loss: {total_loss / len(data_loader)}")

model_save_path = "faster_rcnn_model.pth"

# Тренировка модели
for epoch in range(num_epochs):
    train_one_epoch(model, optimizer, train_loader, device, epoch)
    evaluate(model, val_loader, device)  # Оценка на валидационном наборе

    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved after epoch {epoch + 1} at {model_save_path}")

# Оценка модели на тестовых данных
model.eval()
all_predictions = []
with torch.no_grad():
    for images, targets in test_loader:
        images = list(image.to(device) for image in images)
        predictions = model(images)
        all_predictions.extend(predictions)

print("Model evaluation completed.")