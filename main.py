import os
import random
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

data_dir = './dataset/Images'
annotations_dir = './dataset/Annotations'

#regular values
train_split = 0.7
val_split = 0.15
test_split = 0.15

#fixing of shuffle order
random.seed(42)

#IDEA: when the model will work: detect classes of images and generate annotations


# checks if annotation for images exists, othervise image is not labeled and doesnt fit for steps before the real expluatation
def validate_labels(images, annotations_dir):
    valid_images = []
    black_list = []
    
    for image_file in images:
        annotation_file = image_file.replace('.jpg', '.xml')
        annotation_path = os.path.join(annotations_dir, annotation_file)
        
        
        if os.path.exists(annotation_path):
            valid_images.append(image_file)
        else:
            black_list.append(os.path.join(data_dir, image_file))
    
    return valid_images, black_list

#it gets all images list
images = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]


valid_images, black_list = validate_labels(images, annotations_dir)

random.shuffle(valid_images)  #случайное перемешивание

train_size = int(train_split * len(valid_images))
val_size = int(val_split * len(valid_images))

train_images = valid_images[:train_size]
val_images = valid_images[train_size:train_size + val_size]
test_images = valid_images[train_size + val_size:]


print("Black list (no labels):", len(black_list))
#visualisation of classes distribution

