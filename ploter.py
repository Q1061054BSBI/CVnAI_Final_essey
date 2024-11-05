import os
import matplotlib.pyplot as plt
from collections import Counter
import cv2
import random
import matplotlib.patches as patches


def count_classes(annotations_dir, class_dict):
    id_to_class = {v: k for k, v in class_dict.items()}
    class_counts = Counter()

    for annotation_file in os.listdir(annotations_dir):
        if annotation_file.endswith('.txt'):
            annotation_path = os.path.join(annotations_dir, annotation_file)

            with open(annotation_path, 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    if len(parts) > 0:
                        class_id = int(parts[0])
                        class_name = id_to_class.get(class_id, 'unknown')
                        class_counts[class_name] += 1

    return class_counts

def plot_class_distribution(class_counts, title="Class Distribution"):
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    plt.figure(figsize=(15, 8)) 
    bars = plt.bar(classes, counts, color='skyblue')
    plt.xlabel("Classes")
    plt.ylabel("Number of Objects")
    plt.title(title)
    plt.xticks(rotation=90, ha='right') 
    plt.tight_layout()

    for bar, count in zip(bars, counts):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, int(count), 
                 ha='center', va='bottom', fontsize=10, color='black')

    plt.show()


def visualize_random_augmented_image(augmented_images_dir, augmented_labels_dir, class_dict):
    image_files = [f for f in os.listdir(augmented_images_dir) if f.endswith('.jpg')]
    if not image_files:
        print("No images to show")
        return
    
    random_image_path = os.path.join(augmented_images_dir, random.choice(image_files))
    image_name = os.path.basename(random_image_path)
    label_path = os.path.join(augmented_labels_dir, image_name.replace('.jpg', '.txt'))

    if not os.path.exists(label_path):
        print(f"No annotation found for image {random_image_path}")
        return

    image = cv2.imread(random_image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(1, figsize=(12, 12))
    ax.imshow(image)

    with open(label_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            class_id = int(parts[0])
            class_name = [k for k, v in class_dict.items() if v == class_id][0]
            x_center, y_center, width, height = map(float, parts[1:])

            img_h, img_w = image.shape[:2]
            xmin = int((x_center - width / 2) * img_w)
            ymin = int((y_center - height / 2) * img_h)
            xmax = int((x_center + width / 2) * img_w)
            ymax = int((y_center + height / 2) * img_h)

            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(xmin, ymin - 10, class_name, color='red', fontsize=12, weight='bold')

    plt.axis('off')
    plt.show()

