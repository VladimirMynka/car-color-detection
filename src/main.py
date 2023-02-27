import pandas as pd

from constants import COLORS, DATA_PATH
from dataLoader import load_test, load_test_as_dict, load_train
from evaluater import Evaluater
from models.lotsOfMeans import LotsOfMeans as Model
from processor import Processor


p = Processor()
model = Model(COLORS, p)
train = load_train()[0]
test = load_test_as_dict()
test_list = load_test()[0]

model.fit(train)

test_preds = model.predictForDict(test)

ev = Evaluater(test_preds)
print(ev.reportClasses().to_markdown())
print(ev.reportCommon().to_markdown())

test_preds = model.predict(test_list)

pd.Series(test_preds).to_csv(DATA_PATH / 'preds.csv', index=False, header=False)