from model import Model
from multimodelVariants import fourChildrenMeansRoot as models

class MultiModel(Model):
    def __init__(self, classes, processor=None, len_searcher=None, models=models):
        super().__init__(classes, None, None)
        self.modelSchemes = models
        self.models = {
            key:
            models[key]['modelClass'](
                models[key]['classes'], 
                models[key]['processor'], 
                **models[key]['kwargs']
            ) for key in models
        }

    def reformat_data(self, model_id, data):
        new_data = data.copy()
        unions = self.modelSchemes[model_id]['union']
        classes = self.modelSchemes[model_id]['classes']
        for color in unions:
            new_list = []
            for color2 in unions[color]:
                if color2 in new_data:
                    new_list += new_data[color2]
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

    def load_from_dict(self, dictio):
        for model_id in self.models:
            self.models[model_id].load_from_dict(dictio[model_id])

    def __dict__(self):
        return {model_id: self.models[model_id].__dict__ for model_id in self.models}

