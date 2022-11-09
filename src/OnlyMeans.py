from Model import Model
from tqdm import tqdm
import numpy as np

from Editor import *


class OnlyByMedians(Model):
    def __init__(self, classes):
        super().__init__(classes)
        self.means = {key: np.random.rand(3) for key in classes}
        self.mean = np.array(list(self.means.values())).mean(axis=0)
        for key in self.means:
            self.means[key] -= self.mean

    def fit(self, data):
        data = {key: process_images(data[key]) for key in data}
        m = {color: get_avg_colors(data[color]) for color in tqdm(data)}
        mean = np.mean(list(m.values()), axis=0)
        m = {color: m[color] - mean for color in m}
        self.means = m
        self.mean = mean

    def predictOne(self, image, top=1, metric=cosine):
        processed = process_image(image)
        rgb = get_avg_colors([processed])
        rgb -= self.mean
        dists = [metric(rgb, self.means[key]) for key in self.classes]
        inds = np.argsort(dists)[::-1][:top]
        return {self.classes[ind]: dists[ind] for ind in inds}
