# Environmental Monitoring Object Detection with YOLOv5

This is an educational project in which a computer vision model was developed for object detection on images. The model can be used for preliminary analysis in satellite imagery analysis systems for environmental monitoring or for other purposes where medium detection accuracy is acceptable. The project is based on a pre-trained YOLOv5s model.

## Table of Contents
1. [Requirements](#requirements)
2. [Dataset](#dataset)
3. [Usage](#usage)
4. [Expected Results](#expected-results)

## Requirements
- **Python 3.12**
- Approximately **11.5 GB** of disk space
- **Internet access** (required for certain installations and dataset downloads)

### Required Libraries
Ensure you have the following libraries installed:

- `torch`
- `opencv-python`
- `Pillow`
- `albumentations`
- `ultralytics`
- `certifi`
- `yaml`
- `matplotlib`

Install additional libraries via `pip`:
```bash
pip install torch opencv-python Pillow albumentations ultralytics certifi pyyaml matplotlib
```

## Dataset
The project uses **Pascal VOC 2012**. To use the dataset:

1. Download and extract **Pascal VOC 2012** from [here](http://host.robots.ox.ac.uk/pascal/VOC/).
2. Move the `Annotations` and `JPEGImages` folders into a folder named `datasets` created at the root of the project.

## Usage

### First-Time Run
If you're running the project for the first time:
- Run the `main.py` file to initiate the entire pipeline.

### Subsequent Runs
For subsequent runs, you can choose between the following options:

- **To regenerate the dataset for the model**: 
  - Delete the `resized` folder in the `datasets` directory.
  - Run `generate_data_for_model.py` to preprocess and prepare the dataset again.

- **If the dataset is already generated**:
  - Simply run `model.py` to train or evaluate the model with the existing data.

**Important Notes**:
- Running `main.py` again requires data cleanup by deleting the `resized` folder.
- During execution, two diagram windows will display during the initial data preprocessing phase (approximately the first 5 minutes).
- Process updates and progress information will be printed to the console throughout the execution.

## Expected Results

Approximate expected metrics:
- **mAP@.5**: 0.55
- **mAP@.5:.95**: 0.40
- **Precision**: 54%
- **Recall**: 54%

Additional details:
- **Time for data preprocessing**: ~5 minutes
- **Model training time**: ~13 hours with current parameters
- **Evaluation time on validation and test sets**: ~50 minutes

Testing was conducted on a **MacBook Air M1** with **16GB RAM**, **macOS 15.1**, and **8 cores**.

These results, while moderate, are considered suitable for initial exploration. For more demanding applications, additional training with more data and potentially a more complex model (like YOLOv5m or YOLOv5l) would be recommended.
