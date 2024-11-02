import os
import yaml
from processor import create_class_dict

def generate_config(base_dir, annotations_dir, output_config_path):
    """
    Генерирует YAML конфигурационный файл для YOLO с динамическим созданием имен классов.
    
    Параметры:
    - base_dir (str): Путь к папке с данными (например, папка с ресайзнутыми изображениями).
    - annotations_dir (str): Путь к папке с аннотациями XML для генерации имен классов.
    - output_config_path (str): Путь для сохранения YAML файла конфигурации.
    """
    # Создаем словарь классов
    class_dict = create_class_dict(annotations_dir)
    class_names = [name for name in class_dict.keys()]
    
    # Структура конфигурации
    #Temporary small_train (for tests)
    config = {
        'path': base_dir,  # Путь к папке с изображениями
        'train': 'small_train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': len(class_names),  # Количество классов
        'names': class_names
    }
    
    # Сохранение конфигурации в YAML файл
    with open(output_config_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)

    print(f"YAML configuration file '{output_config_path}' created successfully.")


generate_config('resized', './datasets/Annotations', 'dataset_config.yaml')
