import pickle

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

from models.model import Model
from processor import Processor
from utils import cosine, get_avg_colors_one, get_only_not_black, get_all
from constants import MODELS_PATH
from datetime import datetime


class UseDecisionTree(Model):
    def __init__(self, classes, processor, depth=5, len_searcher=get_only_not_black):
        super().__init__(classes, processor, len_searcher)
        self.clf = DecisionTreeClassifier(max_depth=depth)

    def fit(self, data):
        if self.processor is not None:
            processed = {key: self.processor.process_images(data[key]) for key in tqdm(data)}
        else:
            processed = data
        self.classes = np.array(list(data.keys()))
        X = []
        y = []
        for key in tqdm(data):
            X += [
                list(get_avg_colors_one(p_image, self.len_searcher)) + 
                [get_avg_colors_one(image, self.len_searcher).sum()] 
                for p_image, image in zip(processed[key], data[key])
            ]
            y += [key] * len(data[key])
        self.clf.fit(X, y)

    def predictOne(self, image, top=1, metric=cosine, logging=False):
        if self.processor is not None:
            processed = self.processor.process_image(image, logging=logging)
        else:
            processed = image
        x = list(get_avg_colors_one(processed, self.len_searcher)) + [get_avg_colors_one(image, self.len_searcher).sum()]
        predict = self.clf.predict_proba([x])[0]
        tops = predict.argsort()[::-1][:top]
        return {self.classes[i]:predict[i] for i in tops}

    def load_from_dict(self, dictio):
        self.classes = np.array(dictio['classes'])
        self.processor = Processor.load(dictio['processor']) if dictio['processor'] else None
        self.len_searcher = get_all if dictio['len_searcher'] == 'all' else get_only_not_black
        with open(dictio['tree'], 'rb') as f:
            self.clf = pickle.load(f)

    def __dict__(self):
        (MODELS_PATH / 'trees').mkdir(exist_ok=True)
        name = MODELS_PATH / 'trees' / f'{datetime.now().strftime("%Y_%m_%d %H-%M-%S")}.pickle'
        with name.open('wb') as tree:
            pickle.dump(self.clf, tree)
        return {
            'classes': list(self.classes),
            'processor': self.processor.__dict__() if self.processor is not None else None,
            'len_searcher': 'all' if self.len_searcher == get_all else 'not_black',
            'tree': str(name),
        }
