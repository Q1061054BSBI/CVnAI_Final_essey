import os
import cv2
from resizer import resize_image_and_annotations
from divider import split_dataset
from processor import create_class_dict
from ploter import count_classes, plot_class_distribution
from balancer import augment_minor_classes

os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'

cv2.setLogLevel(3) 

base_dir = './datasets'

print ("Resizing all images to 640x640")
resize_image_and_annotations(f"{base_dir}/JPEGImages", f"{base_dir}/Annotations", f"{base_dir}/resized")

print ("Deviding into train test val & small tain (for bug fixing)")
split_dataset(f"{base_dir}/resized", f"{base_dir}/resized")

# repair_images('./datasets/resized/small_train/images');

print ("Balansing")

class_dict = create_class_dict("./datasets/Annotations")

train_dir = './datasets/resized/train'
current_classes = count_classes(f"{train_dir}/labels", class_dict)

plot_class_distribution(class_counts=current_classes, title="Train set distribution")


images_dir = './datasets/resized/train/images'
labels_dir = './datasets/resized/train/labels'
output_dir = './datasets/resized/train'


lower_threshold = 600 
upper_threshold = 1500 


augment_minor_classes(images_dir, labels_dir, class_dict, lower_threshold, upper_threshold, output_dir)

classes_val = count_classes(f"{train_dir}/labels", class_dict=class_dict)

plot_class_distribution(classes_val, "Destribution of train set after augumaentation images");



