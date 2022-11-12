import numpy as np
from tqdm import tqdm

from model import Model
from processor import Processor
from utils import cosine, get_avg_colors, get_avg_colors_one


class ByOnlyMeans(Model):
    def __init__(self, classes, processor):
        super().__init__(classes, processor)
        self.means = {key: np.random.rand(3) for key in classes}
        self.mean = np.array(list(self.means.values())).mean(axis=0)
        for key in self.means:
            self.means[key] -= self.mean

    def fit(self, data):
        if self.processor is not None:
            data = {key: self.processor.process_images(data[key]) for key in tqdm(data)}
        m = {color: get_avg_colors(data[color]) for color in tqdm(data)}
        mean = np.mean(list(m.values()), axis=0)
        m = {color: m[color] - mean for color in m}
        self.means = m
        self.mean = mean

    def predictOne(self, image, top=1, metric=cosine, logging=False):
        if self.processor is not None:
            image = self.processor.process_image(image, logging=logging)
        rgb = get_avg_colors_one(image)
        rgb -= self.mean
        dists = [metric(rgb, self.means[key]) for key in self.classes]
        inds = np.argsort(dists)[::-1][:top]
        return {self.classes[ind]: dists[ind] for ind in inds}

    def load_weights(self, path):
        dictio = super().load_weights(path)
        self.classes = np.array(dictio['classes'])
        self.means = {key: np.array(dictio['means'][key])
                      for key in dictio['means']}
        self.mean = dictio['mean']
        self.processor = Processor.load(dictio['processor']) if dictio['processor'] else None
        
    def __dict__(self):
        return {
            'classes': list(self.classes),
            'mean': list(self.mean),
            'means': {key: list(self.means[key]) for key in self.means},
            'processor': self.processor.__dict__() if self.processor is not None else None
        }
