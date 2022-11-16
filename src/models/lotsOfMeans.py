import numpy as np

from models.model import Model
from utils import get_avgs_colors, get_avg_colors_one, get_only_not_black, cosine
from tqdm import tqdm, trange

def step(avgs):
    mean = avgs.mean(axis=0)
    std = np.sqrt(((avgs - mean) ** 2).mean(axis=0))
    point1 = mean - std
    point2 = mean + std
    mask1 = (((avgs - point1) ** 2).sum(axis=1) < ((avgs - point2) ** 2).sum(axis=1))
    return {
        'mean': mean,
        'std': std if std.sum() != 0 else np.ones(3),
        'new_groups': [avgs[mask1], avgs[~mask1]]
    }


class LotsOfMeans(Model):
    def __init__(self, classes, processor, len_searcher=get_only_not_black, max_depth=4, min_ratio=0.1):
        super().__init__(classes, processor, len_searcher)
        self.max_depth = max_depth
        self.min_ratio = min_ratio
        self.colors_map = {}
        self.means = {}
        self.stds = {}

    def fit(self, data):
        if self.processor is not None:
            data = {key: self.processor.process_images(data[key]) for key in tqdm(data)}
        self.means = {}
        self.stds = {}
        self.colors_map = {}
        for color in tqdm(self.classes):
            cur = []
            length = len(data[color])
            cur.append(get_avgs_colors(data[color], self.len_searcher))
            for i in range(self.max_depth):
                last = cur
                cur = []
                for avgs, j in zip(last, range(len(last))):
                    if (len(avgs) / length) < self.min_ratio:
                        continue
                    next_step = step(avgs)
                    name = f'{color}_{i}_{j}'
                    self.colors_map[name] = color
                    self.means[name] = next_step['mean']
                    self.stds[name] = next_step['std']
                    cur += next_step['new_groups']
        self.mean = np.array(list(self.means.values())).mean(axis=0)
        self.means = {color: (self.means[color] - self.mean) / self.stds[color] for color in self.means}
        self.classes = list(self.colors_map.keys())

    def predictOne(self, image, top, logging=False, metric=cosine):
        if self.processor is not None:
            image = self.processor.process_image(image, logging=logging)
        rgb = get_avg_colors_one(image, self.len_searcher)
        rgb -= self.mean
        dists = [metric(rgb / self.stds[key], self.means[key]) for key in self.classes]
        inds = np.argsort(dists)[::-1]
        goods = {}
        first_prob = dists[inds[0]]
        for ind in inds:
            real_color = self.colors_map[self.classes[ind]]
            if real_color in goods:
                continue
            else:
                goods[real_color] = dists[ind]
                if len(goods.keys()) == top:
                    break
        max_value = max(goods.values())
        goods = {key: goods[key] / max_value * first_prob for key in goods}
        return dict(sorted(goods.items(), key=lambda elem: elem[1])[::-1])

    def load_from_dict(self, dictio):
        return super().load_from_dict(dictio)

    def __dict__(self):
        return super().__dict__()