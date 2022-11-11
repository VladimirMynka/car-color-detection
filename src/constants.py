import os
from pathlib import Path

PROJECT_PATH = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_PATH / 'data'
TRAIN_PATH = DATA_PATH / 'train'
TEST_PATH = DATA_PATH / 'test'

SHUFFLE_RANDOM_SEED = 1

MODELS_PATH = PROJECT_PATH / 'models'

COLORS = os.listdir(TRAIN_PATH)

KERNELS = [
    [
        [3, 0, -3],
        [10, 0, -10],
        [3, 0, -3]
    ],
    [
        [3, 10, 3],
        [0, 0, 0],
        [-3, -10, -3]
    ]
]
