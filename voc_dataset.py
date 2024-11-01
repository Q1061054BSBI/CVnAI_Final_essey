import os
import xml.etree.ElementTree as ET
import torch
from PIL import Image
from torch.utils.data import Dataset
from data_processing import get_person_label

class VOCDataset(Dataset):
    def __init__(self, images, annotations_dir, data_dir, transforms=None):
        self.images = images
        self.annotations_dir = annotations_dir
        self.data_dir = data_dir
        self.transforms = transforms
        self.unique_classes = self.get_unique_classes(annotations_dir, images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.data_dir, img_name)
        annotation_path = os.path.join(self.annotations_dir, img_name.replace('.jpg', '.xml'))
        
        # Загружаем изображение
        img = Image.open(img_path).convert("RGB")
        
        # Парсим XML-аннотацию
        boxes, labels = [], []
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
            #         # Проверка наличия активных действий, если pose не задан
            #         actions = obj.find('actions')
            #         action_found = False
            #         if actions is not None:
            #             for action in actions:
            #                 if int(action.text) == 1:
            #                     label = f"person_{action.tag.lower()}"
            #                     action_found = True
            #                     break
            #         if not action_found:
            #             label = "person_unspecified"  # Если действия нет, помечаем как "person_unspecified"
            
            # Добавляем индекс класса для каждого объекта
            class_idx = self.unique_classes.index(label)
            labels.append(class_idx)
            
            # Извлекаем координаты bounding box
            bndbox = obj.find('bndbox')
            xmin = int(float(bndbox.find('xmin').text))
            ymin = int(float(bndbox.find('ymin').text))
            xmax = int(float(bndbox.find('xmax').text))
            ymax = int(float(bndbox.find('ymax').text))
            boxes.append([xmin, ymin, xmax, ymax])
        
        # Преобразуем данные в тензоры
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "image_id": image_id, "area": area, "iscrowd": iscrowd}
        
        if self.transforms:
            img = self.transforms(img)

        return img, target

    @staticmethod
    def get_unique_classes(annotations_dir, image_files):
        unique_classes = set()
        
        for image_file in image_files:
            annotation_file = image_file.replace('.jpg', '.xml')
            annotation_path = os.path.join(annotations_dir, annotation_file)
            
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
                #             label = "person_unspecified"  # Если действия нет, помечаем как "person_unspecified"
                
                unique_classes.add(label)
        
        return list(unique_classes)