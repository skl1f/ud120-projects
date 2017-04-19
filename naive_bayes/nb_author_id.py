#!/usr/bin/python

"""
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project.

    Use a Naive Bayes Classifier to identify emails by their authors

    authors and labels:
    Sara has label 0
    Chris has label 1
"""

from time import time
from tools.email_preprocess import preprocess
from sklearn.naive_bayes import GaussianNB

# features_train and features_test are the features for the training
# and testing datasets, respectively
# labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

clf = GaussianNB()

t0 = time()
clf.fit(features_train, labels_train)
print("training time: {0}".format(round(time() - t0, 3), "s"))

t1 = time()
prediction0 = clf.predict(features_test[1])
print("prediction time: {0}".format(round(time() - t1, 3), "s"))
print("prediction: {0}; true result: {1}".format(prediction0, labels_test[1]))

accuracy = clf.score(features_test, labels_test)
print("Accuracy: {0}".format(accuracy))
