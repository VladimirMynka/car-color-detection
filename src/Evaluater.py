import numpy as np
import pandas as pd


class Evaluater:
    def __init__(self, class_predictions):
        self.preds = class_predictions
        self.lengths = {key:len(self.preds[key]) for key in self.preds}
        self.sum_length = np.sum([self.lengths[key] for key in self.preds])

    def recallOneClass(self, class_name):
        real = self.preds[class_name]
        tp = [(1 if class_name in elem else 0) for elem in real]
        return np.sum(tp) / len(tp)

    def precisionOneClass(self, class_name):
        positives = np.sum([np.sum([elem[0] == class_name for elem in self.preds[key]]) for key in self.preds])
        tp = [(1 if elem[0] == class_name else 0) for elem in self.preds[class_name]]
        return np.sum(tp) / positives

    def f1ScoreOneClass(self, class_name):
        recall = self.recallOneClass(class_name)
        precision = self.precisionOneClass(class_name)
        return 2 * recall * precision / (recall + precision)

    def metrics(self, metric_function):
        return {key:metric_function(key) for key in self.preds}

    def recalls(self):
        return self.metrics(self.recallOneClass)

    def precisions(self):
        return self.metrics(self.precisionOneClass)

    def f1Scores(self):
        return self.metrics(self.f1ScoreOneClass)

    def metric(self, metrics_function):
        metrics = metrics_function()
        sum = np.sum([metrics[key] * self.lengths[key] for key in metrics])
        return sum / self.sum_length

    def recall(self):
        return self.metric(self.recalls)

    def precision(self):
        return self.metric(self.precisions)

    def f1ScoreMicro(self):
        return self.metric(self.f1Scores)

    def f1ScoreMacro(self):
        recall = self.recall()
        precision = self.precision()
        return 2 * recall * precision / (recall + precision)

    def reportClasses(self):
        return pd.DataFrame({
            'recall': self.recalls(),
            'precision': self.precisions(),
            'f1Score': self.f1Scores(),
            'count': self.lengths
        })

    def reportCommon(self):
        return pd.DataFrame({
            'recall': [self.recall()],
            'precision': [self.precision()],
            'f1ScoreMicro': [self.f1ScoreMicro()],
            'f1ScoreMacro': [self.f1ScoreMacro()]
        }, index=['all_classes'])