import os
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter
import xml.etree.ElementTree as ET

def visualize_image_and_tensor(image_path):
    # Загружаем изображение
    image = Image.open(image_path).convert("RGB")
    
    # Преобразуем изображение в тензор
    transform = transforms.ToTensor()
    image_tensor = transform(image)
    
    # Визуализация
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    # Исходное изображение
    ax[0].imshow(image)
    ax[0].set_title("Original Image")
    ax[0].axis("off")
    
    # Тензорное представление изображения
    ax[1].imshow(image_tensor.permute(1, 2, 0))
    ax[1].set_title("Image as Tensor")
    ax[1].axis("off")
    
    plt.show()

def validate_labels(images, annotations_dir, data_dir):
    valid_images = []
    black_list = []
    
    for image_file in images:
        annotation_file = image_file.replace('.jpg', '.xml')
        annotation_path = os.path.join(annotations_dir, annotation_file)
        
        if os.path.exists(annotation_path):
            valid_images.append(image_file)
        else:
            black_list.append(os.path.join(data_dir, image_file))
    
    return valid_images, black_list

def count_classes(annotations_dir, image_files):
    class_counts = Counter()
    
    for image_file in image_files:
        annotation_file = image_file.replace('.jpg', '.xml')
        annotation_path = os.path.join(annotations_dir, annotation_file)
        
        if not os.path.exists(annotation_path):
            continue
        
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        
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
                
            class_counts[label] += 1

    return class_counts

def plot_class_distribution(class_counts, title="Class Distribution"):
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    plt.figure(figsize=(15, 8))  # Увеличиваем размер графика
    bars = plt.bar(classes, counts, color='skyblue')
    plt.xlabel("Classes")
    plt.ylabel("Number of Objects")
    plt.title(title)
    plt.xticks(rotation=90, ha='right')  # Поворачиваем подписи на 45 градусов и выравниваем вправо
    plt.tight_layout()

    # Добавление значений на вершину каждого столбца
    for bar, count in zip(bars, counts):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, int(count), 
                 ha='center', va='bottom', fontsize=10, color='black')

    plt.show()

def filter_person_and_other_images(annotations_dir, image_files):
    person_only_images = []
    other_images = []
    
    for image_file in image_files:
        annotation_file = image_file.replace('.jpg', '.xml')
        annotation_path = os.path.join(annotations_dir, annotation_file)
        
        if not os.path.exists(annotation_path):
            continue
        
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        
        only_person = True
        
        for obj in root.findall('object'):
            label = get_person_label(obj)
            # label = obj.find('name').text
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
            if "person" not in label:
                only_person = False
                break
        
        if only_person:
            person_only_images.append(image_file)
        else:
            other_images.append(image_file)
    
    return person_only_images, other_images

def count_poses(annotations_dir, image_files):
    pose_counts = Counter()
    
    for image_file in image_files:
        annotation_file = image_file.replace('.jpg', '.xml')
        annotation_path = os.path.join(annotations_dir, annotation_file)
        
        if not os.path.exists(annotation_path):
            continue
        
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        
        for obj in root.findall('object'):
            if obj.find('name').text == "person":
                label = get_person_label(obj)
            pose_counts[label] += 1

    return pose_counts

def plot_pose_distribution(pose_counts, title="Pose Distribution"):
    poses = list(pose_counts.keys())
    counts = list(pose_counts.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(poses, counts, color='lightcoral')
    plt.xlabel("Pose")
    plt.ylabel("Number of Objects")
    plt.title(title)
    plt.xticks(rotation=90)
    plt.tight_layout()
    
    for bar, count in zip(bars, counts):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, int(count), 
                 ha='center', va='bottom', fontsize=10, color='black')
    
    plt.show()


def get_person_label(obj):
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
    
