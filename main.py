import os
import cv2
from resizer import resize_image_and_annotations
from divider import split_dataset
from processor import create_class_dict
from ploter import count_classes, plot_class_distribution
from balancer import augment_minor_classes
from collections import Counter
from resizer import undersample_dataset
from ultralytics import YOLO

import ssl
import certifi
import warnings

warnings.filterwarnings("ignore")
ssl._create_default_https_context = ssl._create_unverified_context 
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'
cv2.setLogLevel(3) 

base_dir = './datasets'

print ("Resizing all images to 640x640")
resize_image_and_annotations(f"{base_dir}/JPEGImages", f"{base_dir}/Annotations", f"{base_dir}/resized")

print ("Deviding into train test val & small tain (for bug fixing)")
split_dataset(f"{base_dir}/resized", f"{base_dir}/resized")


print ("Balansing")

class_dict = create_class_dict("./datasets/Annotations")

train_dir = './datasets/resized/train'
current_classes = count_classes(f"{train_dir}/labels", class_dict)

plot_class_distribution(class_counts=current_classes, title="Train set distribution")


#augmentation
# images_dir = './datasets/resized/train/images'
# labels_dir = './datasets/resized/train/labels'
# output_dir = './datasets/resized/train'


# lower_threshold = 600 
# upper_threshold = 1500 


# augment_minor_classes(images_dir, labels_dir, class_dict, lower_threshold, upper_threshold, output_dir)

# classes_val = count_classes(f"{train_dir}/labels", class_dict=class_dict)

# plot_class_distribution(classes_val, "Destribution of train set after augumaentation images");


print("undersampling")

def convert_counts_to_id_format(class_to_id, label_counts):
    class_id_counts = Counter({class_to_id[class_name]: label_counts[class_name] 
                               for class_name in label_counts if class_name in class_to_id})
    return class_id_counts



class_id_counts = convert_counts_to_id_format(class_dict, current_classes)

train_images_dir = './datasets/resized/train/images'
train_labels_dir = './datasets/resized/train/labels'
output_images_dir = './datasets/resized/undersampled_train/images'
output_labels_dir = './datasets/resized/undersampled_train/labels'


undersample_dataset(
    images_dir=train_images_dir,
    labels_dir=train_labels_dir,
    output_images_dir=output_images_dir,
    output_labels_dir=output_labels_dir,
    class_counter=class_id_counts,
    max_samples_per_class=1000 
)

current_classes = count_classes(output_labels_dir, class_dict)

plot_class_distribution(class_counts=current_classes, title="Undersampled Train set distribution")

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
