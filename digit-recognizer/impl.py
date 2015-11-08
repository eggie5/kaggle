import pandas as pd
import numpy as np
import pylab as pl
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt


df = pd.read_csv("train.csv")


features = np.array(df.ix[:,1:])
targets = np.array(df["label"])


features_train, features_test, targets_train, targets_test = train_test_split(features,targets, test_size=0.4, random_state=4)

features_train = features_train[:len(features_train)/100]
targets_train = targets_train[:len(targets_train)/100]

# print(targets[3])
# plt.imshow(features[3].reshape(28, 28),cmap=plt.cm.binary)
# plt.show()


clf = SVC(gamma=0.001)

print "training start..."
clf = clf.fit(features_train, targets_train)
print "trainging end"

print clf.score(features_test, targets_test)

print "done"