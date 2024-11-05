import os
import shutil
import cv2
import numpy as np
import albumentations as A
from ploter import count_classes


def augment_image(image_path, bboxes):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Loaging error: {image_path}")

    try:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        raise ValueError(f"Converting to RGB error: {image_path}, {e}")
    
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=0.5),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['labels']))

    labels = [0] * len(bboxes) 
    try:
        transformed = transform(image=image, bboxes=bboxes, labels=labels)
    except Exception as e:
        raise ValueError(f"Transform image error: {image_path}, {e}")

    return transformed['image'], transformed['bboxes']



def augment_minor_classes(images_dir, labels_dir, class_dict, lower_threshold, upper_threshold, output_dir, max_samples_per_class=1000):

    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)

    class_counts = count_classes(labels_dir, class_dict)

    minor_classes = {cls: count for cls, count in class_counts.items() if count < lower_threshold}
    dominant_classes = {cls for cls, count in class_counts.items() if count > upper_threshold}

    print(f"Minor classes: {minor_classes}")
    print(f"Dominant classes: {dominant_classes}")

    for label_file in os.listdir(labels_dir):
        label_path = os.path.join(labels_dir, label_file)
        image_name = label_file.replace('.txt', '.jpg')
        image_path = os.path.join(images_dir, image_name)

        with open(label_path, 'r') as file:
            lines = file.readlines()
            image_bboxes = []
            contains_minor_class = False
            classes_in_image = set()

            #limits checking
            for line in lines:
                parts = line.strip().split()
                class_id = int(parts[0])
                class_name = [k for k, v in class_dict.items() if v == class_id][0]

                if class_name in minor_classes:
                    contains_minor_class = True
                classes_in_image.add(class_name)

                x_center, y_center, width, height = map(float, parts[1:])
                image_bboxes.append([x_center, y_center, width, height])

            #limits checking stage 2
            if contains_minor_class and not any(cls in dominant_classes for cls in classes_in_image):
                if all(class_counts[class_name] + len([bbox for bbox in image_bboxes if [k for k, v in class_dict.items() if v == class_id][0] == class_name]) <= max_samples_per_class for class_name in classes_in_image):
                    #generation images
                    for i in range(2): #amount increasing: 200% 
                        augmented_image, augmented_bboxes = augment_image(image_path, image_bboxes)

                        augmented_image_path = os.path.join(output_dir, 'images', f"aug_{i}_{image_name}")
                        cv2.imwrite(augmented_image_path, cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR))

                        augmented_label_path = os.path.join(output_dir, 'labels', f"aug_{i}_{label_file}")
                        with open(augmented_label_path, 'w') as out_file:
                            for bbox in augmented_bboxes:
                                out_file.write(f"{class_id} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")

    print("Augmentation is complete")




