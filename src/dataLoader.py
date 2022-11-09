import os

from PIL import Image
from tqdm import tqdm

from constants import TEST_PATH, TRAIN_PATH


def load_dir(dir):
    names = os.listdir(dir)
    return [Image.open(name) for name in names], names


def load_train():
    colors = os.listdir(TRAIN_PATH)
    return {color: load_dir(TRAIN_PATH / color) for color in tqdm(colors)}


def load_test():
    return load_dir(TEST_PATH)
