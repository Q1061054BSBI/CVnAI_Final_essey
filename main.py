# main.py
import os
import random
import torch
import torchmetrics
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
from sklearn.metrics import precision_score, recall_score, precision_recall_curve, average_precision_score
from torchvision.ops import box_iou
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


import ssl
import certifi
import warnings


warnings.filterwarnings("ignore")

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
print("I.Cleaning of the dataset from not labled items\n")
valid_images, black_list = validate_labels(images, annotations_dir, data_dir)
print("Black list (no labels):", len(black_list))

print("II.Resizing images and updating of annotations\n")
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

# for index in range(10):
#     visualize_random_augmented_image(resized_images_dir, resized_annotations_dir)

#after resising the resized dirs become new defaut dirs
data_dir = resized_images_dir
annotations_dir = resized_annotations_dir

print("III.Deviding data to train, test and validation sets\n")
random.shuffle(valid_images)

train_size = int(train_split * len(valid_images))
val_size = int(val_split * len(valid_images))
train_images = valid_images[:train_size]
val_images = valid_images[train_size:train_size + val_size]
test_images = valid_images[train_size + val_size:]

print("III*.Deviding a dominant class 'persons' to subclasses with poses and actions\n")
pose_counts = count_poses(annotations_dir, train_images)
# plot_pose_distribution(pose_counts, title="Pose and actions Distribution in Training Dataset")

train_class_counts = count_classes(annotations_dir, train_images)
# plot_class_distribution(train_class_counts, title="Class Distribution in Training Set")

# validation_class_counts = count_classes(annotations_dir, val_images)
# plot_class_distribution(validation_class_counts, title="Class Distribution in Validation Set")

print("IV.Augmentig of minor calsses\n")
#detecting classed from train dataset wich need to be augmented
minor_classes = identify_minor_classes(train_class_counts, threshold=500)
domenant_classes = identify_domenant_classes(train_class_counts, threshold=1500)
augment_voc_classes(classes=minor_classes, image_names=train_images, images_dir=data_dir,  annotations_dir=annotations_dir, output_dir='./resized_dataset', target_multiplier=2, domenant_classes=domenant_classes)

# Обновляем train_images, чтобы включить новые аугментированные изображения
augmented_images = [f for f in os.listdir(data_dir) if '_aug_' in f and f.endswith('.jpg')]
train_images.extend(augmented_images)  # Добавляем аугментированные изображения в список тренировочных данных

#check distribution one morw time
train_class_counts = count_classes(annotations_dir, train_images)
# plot_class_distribution(train_class_counts, title="Class Distribution in Training Set after augmentation")

# visalisation
# for index in range(10):
#     visualize_random_augmented_image(data_dir, annotations_dir)

print("V.Creating waits for better balancing\n")
sampler = create_weighted_sampler(train_images, annotations_dir)

print("VI.Creating objects form sets valid for adding to model\n")
# Создаем DataLoader для каждой выборки
train_dataset = VOCDataset(train_images, annotations_dir, data_dir, transforms=F.to_tensor)
val_dataset = VOCDataset(val_images, annotations_dir, data_dir, transforms=F.to_tensor)
test_dataset = VOCDataset(test_images, annotations_dir, data_dir, transforms=F.to_tensor)

def custom_collate_fn(batch):
    batch = [b for b in batch if b is not None]  # Удаляем пустые элементы
    return tuple(zip(*batch))

train_loader = DataLoader(train_dataset, batch_size=2, sampler=sampler, collate_fn=custom_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=custom_collate_fn)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=custom_collate_fn)

print("VII.Detecting the unique classes list\n")

unique_classes = VOCDataset.get_unique_classes(annotations_dir, valid_images)
num_classes = len(unique_classes) + 1  # +1 для фона

# print("Unique classes:", unique_classes)
print("Number of classes (including background):", num_classes)

print("VIII. Model initialling\n")
# Инициализируем модель Faster R-CNN с предобученной ResNet50
model = fasterrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Установка устройства
device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
model.to(device)

# Оптимизатор
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0005)




def compute_iou(box1, box2):
    """Функция для вычисления Intersection over Union (IoU) между двумя bounding boxes"""
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2

    # Вычисляем координаты пересечения
    inter_x1 = max(x1, x1_p)
    inter_y1 = max(y1, y1_p)
    inter_x2 = min(x2, x2_p)
    inter_y2 = min(y2, y2_p)
    
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Вычисляем площади боксов
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_p - x1_p) * (y2_p - y1_p)
    
    # Вычисляем IoU
    iou = inter_area / (box1_area + box2_area - inter_area)
    return iou

def compute_precision_recall_ap(targets, predictions, iou_threshold=0.5):
    """Функция для вычисления mAP, Precision и Recall на основе меток и IoU"""
    true_labels = []
    pred_scores = []
    pred_labels = []

    for target, prediction in zip(targets, predictions):
        target_boxes = target['boxes']
        target_labels = target['labels']
        
        pred_boxes = prediction['boxes']
        pred_confidences = prediction.get('scores', torch.ones(len(pred_boxes)))  # предполагаем 1 для всех при отсутствии
        pred_labels_batch = prediction['labels']
        
        for pred_box, pred_label, pred_score in zip(pred_boxes, pred_labels_batch, pred_confidences):
            matched = False
            for target_box, target_label in zip(target_boxes, target_labels):
                if target_label == pred_label and compute_iou(pred_box, target_box) >= iou_threshold:
                    matched = True
                    true_labels.append(1)  # 1 для верного предсказания
                    break
            if not matched:
                true_labels.append(0)  # 0 для неверного предсказания
            
            pred_scores.append(pred_score)
            pred_labels.append(pred_label)

    precision, recall, _ = precision_recall_curve(true_labels, pred_scores)
    average_precision = average_precision_score(true_labels, pred_scores)

    return average_precision, precision.mean(), recall.mean()


# def get_person_label_from_name(obj):
#     """
#     Функция для обработки подкатегорий класса person с учетом actions и pose.
#     Если label уже является подкатегорией (начинается с 'person_'), возвращает её без изменений.
#     Если label - основной 'person', проверяет actions и pose для определения подкатегории.
    
#     Параметры:
#     - obj (xml.Element): XML-элемент, представляющий объект аннотации.
    
#     Возвращает:
#     - label (str): метка для подкатегорий person или других классов.
#     """
#     label = obj.find('name').text

#     # Если уже является подкатегорией person, возвращаем как есть
#     if label.startswith("person_"):
#         return label

#     # Если label - основной "person", проверяем actions
#     if label == "person":
#         actions = obj.find('actions')
#         action_found = False
#         if actions is not None:
#             all_zero = True  # Флаг, указывающий, что все действия равны нулю
#             for action in actions:
#                 if int(action.text) == 1:
#                     label = f"person_{action.tag.lower()}"
#                     action_found = True
#                     break
#                 if int(action.text) != 0:
#                     all_zero = False
            
#             # Если все действия равны нулю, отмечаем как unspecified
#             if not action_found and all_zero:
#                 label = "person_unspecified"
        
#         # Если action не найден или он unspecified, проверяем pose
#         if not action_found or label == "person_unspecified":
#             pose = obj.find('pose').text if obj.find('pose') is not None else 'Unspecified'
#             if pose != "Unspecified":
#                 label = f"person_{pose.lower()}"
#             else:
#                 label = "person_unspecified"

#     return label

def evaluate(model, data_loader, device):
    """
    Оценивает модель на заданном наборе данных, рассчитывая потери, mAP, точность и полноту.

    Параметры:
    - model: модель для оценки.
    - data_loader: DataLoader для набора данных оценки.
    - device: устройство для выполнения вычислений (CPU или GPU).
    
    Возвращает:
    - Словарь с метриками 'loss', 'mAP', 'precision', 'recall'.
    """
    model.eval()
    total_loss = 0
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for images, targets in data_loader:
            # Подготовка изображений и целей
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Получение предсказаний и расчет потерь
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()

            # Получение предсказаний от модели
            predictions = model(images)

            # Сохранение предсказаний и целевых значений для расчета метрик
            for target, prediction in zip(targets, predictions):
                all_targets.append({
                    'boxes': target['boxes'].cpu(),
                    'labels': target['labels'].cpu()
                })
                all_predictions.append({
                    'boxes': prediction['boxes'].cpu(),
                    'labels': prediction['labels'].cpu(),
                    'scores': prediction['scores'].cpu()
                })

    # Расчет метрик (mAP, точности, полноты)
    mAP, precision, recall = compute_precision_recall_ap(all_targets, all_predictions)

    # Печать метрик
    avg_loss = total_loss / len(data_loader)
    # print(f"Validation Loss: {avg_loss:.4f}")
    # print(f"mAP: {mAP:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

    # Возврат модели в режим обучения
    model.train()

    # Возвращение метрик в виде словаря
    return {'loss': avg_loss, 'mAP': mAP, 'precision': precision, 'recall': recall}
# Функция для обучения одной эпохи
def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    total_loss = 0
    for images, targets in data_loader:
        print("Images type:", type(images))
        print("Targets type:", type(targets))
        print("Length of images:", len(images))
        print("Length of targets:", len(targets))
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        print("Training batch targets:")
        for target in targets:
            print("Boxes:", target.get('boxes'))
            print("Labels:", target.get('labels'))
            print("Image ID:", target.get('image_id'))
            print("Boxes shape:", target.get('boxes').shape if target.get('boxes') is not None else None)
            print("Labels shape:", target.get('labels').shape if target.get('labels') is not None else None)

        # Попытка вызова модели и обработка ошибок
        # try:
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            if device.type == 'mps':
                torch.mps.empty_cache()

            total_loss += losses.item()
        # except Exception as e:
        #     print(f"Error during training: {e}")
        #     continue  # Пропустить цикл, если возникает ошибка

    print(f"Epoch {epoch}, Loss: {total_loss / len(data_loader)}")

model_save_path = "faster_rcnn_model.pth"

print("IX. Model training\n")
for epoch in range(num_epochs):
    print(f"Epoch {epoch +1} started...\n")
    train_one_epoch(model, optimizer, train_loader, device, epoch)

     # Оценка на валидационном наборе
    val_loss, val_mAP, val_precision, val_recall = evaluate(model, val_loader, device)
    print(f"Epoch {epoch + 1} - Validation Loss: {val_loss:.4f}, mAP: {val_mAP:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")
    

    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved after epoch {epoch + 1} at {model_save_path}")


print("IX. Evaluation on test set\n")
test_loss, test_mAP, test_precision, test_recall = evaluate(model, test_loader, device)
print(f"Test Set - Loss: {test_loss:.4f}, mAP: {test_mAP:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")


print("Model evaluation completed.")