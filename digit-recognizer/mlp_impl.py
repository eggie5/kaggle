from opendeep.models import Prototype, Dense, Softmax
from opendeep.models.utils import Noise
from opendeep.optimization.loss import Neg_LL
from opendeep.optimization import AdaDelta
from opendeep.data import MNIST
from theano.tensor import matrix, lvector
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from opendeep.data import Dataset

print "Getting data..."
# data = MNIST()
df = pd.read_csv("train.csv")
features = np.array(df.ix[:,1:])
targets = np.array(df["label"])
features_train, features_test, targets_train, targets_test = train_test_split(features,targets, test_size=0.4, random_state=4)

data = Dataset(features_train, train_targets=targets_train,
                 test_inputs=features_test, test_targets=targets_test)

print "Creating model..."
in_shape = (None, 28*28)
in_var = matrix('xs')
mlp = Prototype()
mlp.add(Dense(inputs=(in_shape, in_var), outputs=512, activation='relu'))
mlp.add(Noise, noise='dropout', noise_level=0.5)
mlp.add(Dense, outputs=512, activation='relu')
mlp.add(Noise, noise='dropout', noise_level=0.5)
mlp.add(Softmax, outputs=10, out_as_probs=False)

print "Training..."
target_var = lvector('ys')
loss = Neg_LL(inputs=mlp.models[-1].p_y_given_x, targets=target_var, one_hot=False)

optimizer = AdaDelta(model=mlp, loss=loss, dataset=data, epochs=10)
optimizer.train()

print "Predicting..."
predictions = mlp.run(data.test_inputs)

print "Accuracy: ", float(sum(predictions==data.test_targets)) / len(data.test_targets)


# now run the dataset from kaggle
test_features = np.array(pd.read_csv("test.csv"))
predictions = mlp.run(test_features)

f = open('mlp_  predictions', 'w')
for i, digit in enumerate(predictions):
    f.write(str(i+1)+","+str(digit)+"\n")
f.close()


print "done"

# 698 .90714
# 666 .93071, +0.02357