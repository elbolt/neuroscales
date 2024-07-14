import yaml


def load_config(config_path: str) -> dict:
    """ Load configuration from YAML file. """
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)