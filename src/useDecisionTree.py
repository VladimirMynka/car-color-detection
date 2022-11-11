import numpy as np
from tqdm import tqdm
from sklearn.tree import DecisionTreeClassifier
import pickle

from model import Model
from processor import Processor
from utils import cosine, get_avg_colors_one


class UseDecisionTree(Model):
    def __init__(self, classes, processor, depth=5):
        super().__init__(classes, processor)
        self.clf = DecisionTreeClassifier(max_depth=depth)

    def fit(self, data):
        processed = {key: self.processor.process_images(data[key]) for key in tqdm(data)}
        X = []
        y = []
        for key in tqdm(data):
            X += [list(get_avg_colors_one(p_image)) + [get_avg_colors_one(image).sum()] for p_image, image in zip(processed[key], data[key])]
            y += [key] * len(data[key])
        self.clf.fit(X, y)

    def predictOne(self, image, top=1, metric=cosine):
        processed = self.processor.process_image(image)
        x = list(get_avg_colors_one(processed)) + [get_avg_colors_one(image).sum()]
        predict = self.clf.predict_proba([x])[0]
        tops = predict.argsort()[::-1][:top]
        return {self.classes[i]:predict[i] for i in tops}

    def load_weights(self, path):
        dictio = super().load_weights(path)
        self.classes = np.array(dictio['classes'])
        self.clf = pickle.loads(dictio['tree'])
        self.processor = Processor.load(dictio['processor'])

    def __dict__(self):
        return {
            'classes': list(self.classes),
            'processor': self.processor.__dict__(),
            'tree': pickle.dumps(self.clf),
        }
