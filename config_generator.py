import os
import yaml
from processor import create_class_dict

def generate_config(base_dir, annotations_dir, output_config_path):

    class_dict = create_class_dict(annotations_dir)
    class_names = [name for name in class_dict.keys()]
    
    config = {
        'path': base_dir,
        'train': 'undersampled_train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': len(class_names),
        'names': class_names
    }
    
    with open(output_config_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)

    print(f"YAML configuration file '{output_config_path}' created successfully.")


generate_config('resized', './datasets/Annotations', 'dataset_config.yaml')
