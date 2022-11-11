import os

from PIL import Image
from tqdm import tqdm

from constants import TEST_PATH, TRAIN_PATH


def load_dir(dir):
    names = os.listdir(dir)
    names = [int(name[:-4]) for name in names]
    names = [f'{name}.jpg' for name in list(sorted(names))]
    return [Image.open(dir / name) for name in names], names


def load_train():
    colors = os.listdir(TRAIN_PATH)
    all = {color: load_dir(TRAIN_PATH / color) for color in tqdm(colors)}
    return (
        {color: all[color][0] for color in colors},
        {color: all[color][1] for color in colors}
    )


def load_test():
    return load_dir(TEST_PATH)
