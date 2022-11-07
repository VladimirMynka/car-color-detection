import random
from tqdm import tqdm
import numpy as np


class Model:
    def __init__(self, classes):
        self.classes = np.array(classes)
    
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


