import os

def get(rel_path):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), rel_path))