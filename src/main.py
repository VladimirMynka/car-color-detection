from dataLoader import load_test, load_train
from constants import COLORS
from processor import Processor
from byOnlyMeans import ByOnlyMeans

from sklearn.tree import DecisionTreeClassifier

processor = Processor()
model = ByOnlyMeans(COLORS, processor)

train_imgs, train_names = load_train()
model.fit(train_imgs)

test_imgs, test_names = load_test()
print(*[model.predictOne(img, top=3) for img in test_imgs[:10]])

model.save()
model.tree