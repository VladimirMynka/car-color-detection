import os

from PIL import Image
from tqdm import tqdm
import pandas as pd

from constants import TEST_PATH, TRAIN_PATH, TEST_VALIDATION
from utils import df_to_dict


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


def load_test_as_dict():
    validation = pd.read_excel(TEST_VALIDATION)
    asdict = df_to_dict(validation)
    return {key:[Image.open(TEST_PATH / name) for name in asdict[key]] for key in asdict}
