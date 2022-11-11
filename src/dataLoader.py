import os
import pandas as pd
from PIL import Image
from tqdm import tqdm

from constants import TEST_PATH, TRAIN_PATH, SHUFFLE_RANDOM_SEED


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


def load_train_dataframe():
    images, _ = load_train()
    df = pd.DataFrame({"Image": [image for group in images.values() for image in group],
                       "Target": [color for color in images.keys() for _ in images[color]]})
    return df.sample(frac=1, random_state=SHUFFLE_RANDOM_SEED).reset_index(drop=True)


def load_test_dataframe():
    return pd.DataFrame({"Image": load_test()}).sample(frac=1, random_state=SHUFFLE_RANDOM_SEED).reset_index(drop=True)
