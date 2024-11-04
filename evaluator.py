import os
import torch
import cv2

class YOLODataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        """
        Класс для загрузки датасета YOLO.
        
        Параметры:
        - images_dir (str): Путь к папке с изображениями.
        - labels_dir (str): Путь к папке с аннотациями.
        - class_dict (dict): Словарь классов с ID.
        - transform (callable, optional): Аугментации для изображений.
        """
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.images_dir, image_name)
        label_path = os.path.join(self.labels_dir, image_name.replace('.jpg', '.txt'))

        # Загрузка изображения
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        bboxes = []
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    labels.append(class_id)
                    x_center, y_center, width, height = map(float, parts[1:])
                    bboxes.append([x_center, y_center, width, height])

        # Применение аугментаций
        if self.transform:
            transformed = self.transform(image=image, bboxes=bboxes, labels=labels)
            image = transformed['image']
            bboxes = transformed['bboxes']

        target = {'boxes': torch.tensor(bboxes, dtype=torch.float32),
                  'labels': torch.tensor(labels, dtype=torch.int64)}
        return image, target

def evaluate_model_metrics(model, class_names):
    results = model.val(split='test', verbose=True)

    # print(f"Overall Precision: {results.metrics.precision:.4f}")
    # print(f"Overall Recall: {results.metrics.recall:.4f}")
    # print(f"Overall mAP@0.5: {results.metrics.map50:.4f}")
    # print(f"Overall mAP@0.5:0.95: {results.metrics.map:.4f}")

    # # Display metrics for each class
    # print("\nClass-wise Metrics:")
    # for idx, class_name in enumerate(class_names):
    #     precision = results.class_metrics['precision'][idx]
    #     recall = results.class_metrics['recall'][idx]
    #     map50 = results.class_metrics['map50'][idx]
    #     map = results.class_metrics['map'][idx]

    #     print(f"{class_name:20} | Precision: {precision:.4f} | Recall: {recall:.4f} | mAP@0.5: {map50:.4f} | mAP@0.5:0.95: {map:.4f}")
        
        

