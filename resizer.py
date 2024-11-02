import os
import xml.etree.ElementTree as ET
import cv2


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

            if not os.path.exists(annotation_path):
                print(f"Аннотация для {image_name} не найдена, пропуск...")
                continue

            # Открываем изображение и изменяем его размер с помощью OpenCV
            img = cv2.imread(image_path)
            if img is None:
                print(f"Ошибка при загрузке изображения {image_name}")
                continue
            original_height, original_width = img.shape[:2]
            img_resized = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)
            output_image_path = os.path.join(output_images_dir, image_name)
            cv2.imwrite(output_image_path, img_resized, [cv2.IMWRITE_JPEG_QUALITY, 95])

            # Обновляем аннотации
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

                # Пересчет координат с учетом нового размера изображения
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

    print("Ресайз изображений и обновление аннотаций завершены.")