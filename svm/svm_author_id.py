#!/usr/bin/python

"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
    Sara has label 0
    Chris has label 1
"""

from time import time
from tools.email_preprocess import preprocess
from sklearn.svm import SVC

# features_train and features_test are the features for the training
# and testing datasets, respectively
# labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

# features_train = features_train[:len(features_train) // 100]
# labels_train = labels_train[:len(labels_train) // 100]

clf = SVC(C=10000.)

t0 = time()
clf.fit(features_train, labels_train)
print("training time: {0}".format(round(time() - t0, 3), "s"))

t1 = time()
prediction0 = clf.predict(features_test)
print("prediction time: {0}".format(round(time() - t1, 3), "s"))
# print("10: {0}, 26: {1}, 50: {2}".format(prediction0[10], prediction0[26], prediction0[50]))
print(sum(prediction0))

accuracy = clf.score(features_test, labels_test)
print("Accuracy: {0}".format(accuracy))
