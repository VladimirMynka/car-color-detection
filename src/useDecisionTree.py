import numpy as np
import pandas as pd
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
        y = data["Target"]
        X = data.drop("Target", axis=1)
        X["processed"] = self.processor.process_images(X["Image"])
        X["average RGB"] = X.apply(
            lambda x: get_avg_colors_one(x["processed"]), axis=1)
        X_final = pd.DataFrame(
            X["average RGB"].tolist(), columns=["R", "G", "B"])
        X_final["brightness"] = X.apply(
            lambda x: get_avg_colors_one(x["Image"]).sum(), axis=1)
        self.clf.fit(X_final, y)

    def predictOne(self, image, top=1, metric=cosine):
        processed = self.processor.process_image(image)
        x = pd.DataFrame([list(get_avg_colors_one(
            processed)) + [get_avg_colors_one(image).sum()]], columns=["R", "G", "B", "brightness"])
        predict = self.clf.predict_proba(x)[0]
        tops = predict.argsort()[::-1][:top]
        return {self.classes[i]: predict[i] for i in tops}

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
