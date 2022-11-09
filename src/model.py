import json
import os
import random
from datetime import datetime

import numpy as np
from tqdm import tqdm

from constants import MODELS_PATH


class Model:
    def __init__(self, classes, processor):
        self.classes = np.array(classes)
        self.processor = processor
    
    def fit(self, data):
        pass

    def predictOne(self, image, top):
        return {random.choice(self.classes):1.0 for _ in range(top)}

    def predict(self, images, top=1):
        return [list(self.predictOne(image, top).keys()) for image in images]

    def predict_for_dict(self, dictio, top=1):
        return {key:self.predict(dictio[key], top) for key in tqdm(dictio)}

    def __call__(self, image, top):
        return self.predict(image, top)

    def save(self, name=''):
        name += f'_{datetime.now().strftime("%Y_%m_%d %H-%M-%S")}.json'
        os.makedirs(MODELS_PATH, exist_ok=True)
        with (MODELS_PATH / name).open('w', encoding='utf-8') as f:
            f.write(json.dumps(self.__dict__(), indent=1))

    def load_weights(self, path):
        with open(path, 'r') as f:
            weights = json.loads(f.read())
        return weights
        
    def __dict__(self):
        return {
            'classes': list(self.classes)
        }
