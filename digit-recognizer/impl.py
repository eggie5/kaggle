import pandas as pd
import numpy as np
import pylab as pl
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt


df = pd.read_csv("train.csv")


features = np.array(df.ix[:,1:])
targets = np.array(df["label"])

test_features = np.array(pd.read_csv("test.csv"))


features_train, features_test, targets_train, targets_test = train_test_split(features,targets, test_size=0.4, random_state=4)

# features_train = features_train[:len(features_train)/100]
# targets_train = targets_train[:len(targets_train)/100]

# print(targets[3])
# plt.imshow(features[3].reshape(28, 28),cmap=plt.cm.binary)
# plt.show()

# ERROR: Column '2' was not expected (Line 1, Column 1)
# ERROR: Required column 'ImageId' could not be found
# ERROR: Required column 'Label' could not be found.


clf = SVC(kernel="linear")

print "training start..."
clf = clf.fit(features_train, targets_train)
print "trainging end"

print clf.score(features_test, targets_test)

print "running predictions..."
predictions = clf.predict(test_features)
print "end predictions"


f = open('predictions', 'w')
for i, digit in enumerate(predictions):
    f.write(str(i+1)+","+str(digit)+"\n")
f.close()

# np.savetxt('predictions', predictions, delimiter='n', fmt="%i")   # X is an array

print "done"