import os
from box.exceptions import BoxValueError
import yaml
from src.textSummarizer.logging import logger
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (Path): path like input

    Raises:
        ValueError: if yaml file is empty
        Exception: if any other error occurs during file reading or parsing

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            if content is None:
                raise ValueError("yaml file is empty")
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        logger.error(f"An unexpected error occurred while reading or parsing YAML file at {path_to_yaml}: {e}")
        raise Exception(f"An unexpected error occurred: {e}")

@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        verbose (bool, optional): log message if directories are created. Defaults to True.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")

# Removed the second definition of read_yaml to avoid confusion and ensure the ConfigBox version is used.
# def read_yaml(path_to_yaml: str):
#     try:
#         with open(path_to_yaml, "r") as yaml_file:
#             content = yaml.safe_load(yaml_file)
#             logger.info(f"yaml file: {path_to_yaml} loaded successfully")
#             return content
#     except Exception as e:
#         raise e