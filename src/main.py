from dataLoader import load_test, load_train
from processor import Processor

p = Processor()
d = p.__dict__()
d['when_black'] = 94
p2 = Processor.load(d)
print(p2.__dict__())