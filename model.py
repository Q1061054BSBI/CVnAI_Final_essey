import ssl
import certifi
import warnings
from ultralytics import YOLO
from processor import create_class_dict


warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="Corrupt JPEG data")
ssl._create_default_https_context = ssl._create_unverified_context 
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

print("Model Training")
#directions
train_dir = './datasets/resized/small_train'
val_dir = './datasets/resized/val'
test_dir = './datasets/resized/test'

#train params
num_epochs = 5
batch_size = 8
learning_rate = 0.001


model = YOLO('yolov5s.pt') 

model.train(
    data='dataset_config.yaml',
    epochs=num_epochs,
    imgsz=640,
    lr0=learning_rate
)

print("Validation of the model on a test set")

model.val(split='test', verbose=True)

