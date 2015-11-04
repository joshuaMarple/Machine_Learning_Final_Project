from __future__ import print_function
import numpy as np
import urllib
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA

from sklearn import cross_validation
import sys

sys.stdout.write("Beginning data set acquisition... ")
sys.stdout.flush()

# training_url = "http://www.ittc.ku.edu/~jhuan/EECS738_F15/slides/EECS738_Train.csv"
# training_raw_data = urllib.urlopen(training_url)
training_raw_data = open("EECS738_Train.csv", "r")
training_dataset = np.loadtxt(training_raw_data, delimiter=",")

# testing_url = "http://www.ittc.ku.edu/~jhuan/EECS738_F15/slides/EECS738_Test.csv"
# testing_raw_data = urllib.urlopen(testing_url)
# testing_dataset = np.loadtxt(testing_raw_data, delimiter=",")

print("complete.")
sys.stdout.write("Loading into datasets... ")
sys.stdout.flush()

X = training_dataset[:,2:10]
Y = training_dataset[:,1]

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=0.2)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
# X_prime = testing_dataset[:,1:51]

print("complete.")

sys.stdout.write("Model fitting... ")
sys.stdout.flush()

models = [KNeighborsClassifier(n_neighbors=3),
          DecisionTreeClassifier(),
          RandomForestClassifier(),
          LDA(),
          GaussianNB()]

model_names = ["K Neighbors", "Decison Tree", "Random Forest", "LDA", "Naive Bayes"]

for i in models:
    i.fit(X_train, y_train)

print("complete.")
sys.stdout.write("Model prediction... ")
sys.stdout.flush()

# predicted = model3.predict(X_prime)
# predicted = model3.predict(X)

predicted_results = [i.predict(X_test) for i in models]

print("complete.")

for predicted, model in zip(predicted_results, model_names):
    print(metrics.classification_report(y_test, predicted))
    print("Accuracy of ", model, ": ", metrics.accuracy_score(y_test, predicted))
    # print(metrics.roc_curve(Y, predicted))

# for model in models:
#     print(model.score(X_test, y_test))

# for model in models:
#     print(cross_validation.cross_val_score(model, X, Y, cv=5))
