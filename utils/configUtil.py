import os
import yaml


class ConfigUtil:
    def __init__(self, file_path):
        self.file_path = file_path
        self.config = None
        self.load_config()

    def load_config(self):
        try:
            with open(self.file_path, 'r', encoding='utf-8') as file:
                self.config = yaml.safe_load(file)
        except FileNotFoundError:
            print(f"Error: Config file '{self.file_path}' not found.")
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file: {e}")

    def __getitem__(self, key):
        if self.config is not None:
            keys = key.split('.')
            current_dict = self.config
            try:
                for k in keys:
                    current_dict = current_dict[k]
                return current_dict
            except KeyError:
                return None
        else:
            return None


current_path = os.path.abspath(".")
yaml_path = os.path.join(current_path, "config.yaml")

# 通过from xxx import CONFIG, 然后像CONFIG['redis.host']这样使用
CONFIG = ConfigUtil(yaml_path)
