import os
from pathlib import Path


def ensure_dir(path):
    """确保目录存在"""
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def get_project_root():
    """获取项目根目录"""
    return Path(__file__).resolve().parent.parent.parent
