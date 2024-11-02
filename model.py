import ssl
import certifi
import warnings
from torch.utils.data import DataLoader
from ultralytics import YOLO
from evaluator import YOLODataset, evaluate_model_metrics
from processor import create_class_dict


warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="Corrupt JPEG data")
ssl._create_default_https_context = ssl._create_unverified_context 
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())


#directions
train_dir = './datasets/resized/small_train'
val_dir = './datasets/resized/val'
test_dir = './datasets/resized/test'

#train params
num_epochs = 1
batch_size = 8
learning_rate = 0.001


# Загрузка модели YOLO
model = YOLO('yolov5s.pt')  # Вы можете заменить 'yolov5s.pt' на другую версию модели (yolov5m.pt, yolov5l.pt и т.д.)

# # Настройка обучения
model.train(
    data='dataset_config.yaml',
    epochs=num_epochs,
    imgsz=640,
    lr0=learning_rate
)

