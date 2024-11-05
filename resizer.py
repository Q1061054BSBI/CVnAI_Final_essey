import os
import xml.etree.ElementTree as ET
from collections import Counter
import cv2
import random
import shutil

def resize_image_and_annotations(input_images_dir, input_annotations_dir, output_dir, target_size=(640, 640)):
    output_images_dir = os.path.join(output_dir, 'JPEGImages')
    output_annotations_dir = os.path.join(output_dir, 'Annotations')
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_annotations_dir, exist_ok=True)

    target_width, target_height = target_size

    for image_name in os.listdir(input_images_dir):
        if image_name.endswith('.jpg'):
            image_path = os.path.join(input_images_dir, image_name)
            annotation_path = os.path.join(input_annotations_dir, image_name.replace('.jpg', '.xml'))

            #excluding not labled images
            if not os.path.exists(annotation_path):
                print(f"Abbotations for {image_name} were not found, passing...")
                continue

            #excluding images with errors
            img = cv2.imread(image_path)
            if img is None:
                print(f"Image Loading error {image_name}")
                continue


            original_height, original_width = img.shape[:2]
            img_resized = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)
            output_image_path = os.path.join(output_images_dir, image_name)
            cv2.imwrite(output_image_path, img_resized, [cv2.IMWRITE_JPEG_QUALITY, 95])

            #Anniataions update
            tree = ET.parse(annotation_path)
            root = tree.getroot()
            scale_x = target_width / original_width
            scale_y = target_height / original_height
            for obj in root.findall('object'):
                bndbox = obj.find('bndbox')
                xmin = int(float(bndbox.find('xmin').text))
                ymin = int(float(bndbox.find('ymin').text))
                xmax = int(float(bndbox.find('xmax').text))
                ymax = int(float(bndbox.find('ymax').text))

                new_xmin = int(xmin * scale_x)
                new_ymin = int(ymin * scale_y)
                new_xmax = int(xmax * scale_x)
                new_ymax = int(ymax * scale_y)

                bndbox.find('xmin').text = str(new_xmin)
                bndbox.find('ymin').text = str(new_ymin)
                bndbox.find('xmax').text = str(new_xmax)
                bndbox.find('ymax').text = str(new_ymax)

            output_annotation_path = os.path.join(output_annotations_dir, image_name.replace('.jpg', '.xml'))
            tree.write(output_annotation_path)

    print("Images resize and annotations update is completed")


def undersample_dataset(images_dir, labels_dir, output_images_dir, output_labels_dir, class_counter, max_samples_per_class=1000):
   
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    sorted_classes = sorted(class_counter.items(), key=lambda x: x[1])

    selected_counts = Counter()

    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    random.shuffle(image_files)

    for class_id, _ in sorted_classes:
        if selected_counts[class_id] >= max_samples_per_class:
            continue 

        for image_file in image_files:
            label_file = image_file.replace('.jpg', '.txt')
            label_path = os.path.join(labels_dir, label_file)
            image_path = os.path.join(images_dir, image_file)

            if not os.path.exists(label_path):
                continue

            with open(label_path, 'r') as f:
                objects_in_image = [int(line.split()[0]) for line in f]
            
            if class_id in objects_in_image:
                if all(selected_counts[obj] < max_samples_per_class for obj in objects_in_image):
                    shutil.copy(image_path, os.path.join(output_images_dir, image_file))
                    shutil.copy(label_path, os.path.join(output_labels_dir, label_file))

                    for obj in objects_in_image:
                        selected_counts[obj] += 1

                    if selected_counts[class_id] >= max_samples_per_class:
                        break

    print(f"Undersampling is complete.")


def convert_counts_to_id_format(class_to_id, label_counts):
    class_id_counts = Counter({class_to_id[class_name]: label_counts[class_name] for class_name in label_counts if class_name in class_to_id})
    return class_id_counts


