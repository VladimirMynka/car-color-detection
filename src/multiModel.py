from model import Model
from byOnlyMeans import ByOnlyMeans
from useDecisionTree import UseDecisionTree
from processor import Processor

models = {
    0: {
        'modelClass': ByOnlyMeans,
        'processor': Processor(when_grey_std=10),
        'goto': {
            'Grey': 1
        },
        'union': {
            'Grey': ['Black', 'Grey', 'White']
        },
        'classes': ['Blue', 'Brown', 'Cyan', 'Green', 'Grey', 'Orange', 'Red', 'Violet', 'Yellow']
    },
    1: {
        'modelClass': ByOnlyMeans,
        'processor': Processor(when_grey_std=-1),
        'goto': {},
        'union': {},
        'classes': ['Black', 'Grey', 'White']
    }
}

class MultiModel(Model):
    def __init__(self, classes, processor=None, models=models):
        super().__init__(classes, None)
        self.modelSchemes = models
        self.models = {key:models[key]['modelClass'](models[key]['classes'], models[key]['processor']) for key in models}

    def reformat_data(self, model_id, data):
        new_data = data.copy()
        unions = self.modelSchemes[model_id]['union']
        classes = self.modelSchemes[model_id]['classes']
        for color in unions:
            new_list = []
            for color2 in unions[color]:
                if color2 in new_data:
                    new_list += new_data[color2]
                    del(new_data[color2])
            new_data[color] = new_list
        colors = list(new_data.keys())
        for color in colors:
            if color not in classes:
                del(new_data[color])
        return new_data

    def fit(self, data):
        for model_id in self.models:
            reform_data = self.reformat_data(model_id, data)
            self.models[model_id].fit(reform_data)
        
    def predictOne(self, image, top=1, logging=False, start_model_id=0):
        model_id = start_model_id
        predict = list(self.models[model_id].predictOne(image, top=1, logging=logging).keys())[0]
        while predict in self.modelSchemes[model_id]['goto']:
            model_id = self.modelSchemes[model_id]['goto'][predict]
            predict = list(self.models[model_id].predictOne(image, top=1, logging=logging).keys())[0]
        return {predict: 1.0}


