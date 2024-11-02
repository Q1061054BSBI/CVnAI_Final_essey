import yaml

# Define the structure of the YAML configuration
config = {
    'path': '',  # Path to the root directory of your dataset
    'train': 'small/train/Images',
    'val': 'val/Images',
    'test': 'test/Images',
    'nc': 36,  # Number of classes
    'names': [
        'person', 'car', 'dog', 'cat', 'bicycle', 'truck', 'bird', 'bus', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
        'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports_ball', 'kite',
        'baseball_bat', 'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket',
        'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon'
    ]
}

# Save the configuration as a YAML file
with open('dataset_config.yaml', 'w') as file:
    yaml.dump(config, file, default_flow_style=False)

print("YAML configuration file 'dataset_config.yaml' created successfully.")
