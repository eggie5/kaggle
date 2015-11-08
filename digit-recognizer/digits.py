import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.cross_validation import train_test_split

digits = datasets.load_digits()


# print(digits.target[2])
# plt.imshow(digits.data[2].reshape(8, 8),cmap=plt.cm.binary)
# plt.show()



features_train, features_test, targets_train, targets_test = train_test_split(digits.data,digits.target, test_size=0.4, random_state=4)

classifier = svm.SVC(gamma=0.001)
classifier = classifier.fit(features_train, targets_train)
print classifier.score(features_test, targets_test)