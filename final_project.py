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

X = training_dataset[:,2:52]
Y = training_dataset[:,2]
# X_prime = testing_dataset[:,1:51]

print("complete.")

sys.stdout.write("Model fitting... ")
sys.stdout.flush()

models = [KNeighborsClassifier(n_neighbors=3),
          DecisionTreeClassifier(),
          RandomForestClassifier(),
          LDA(),
          GaussianNB()]

for i in models:
    i.fit(X,Y)

print("complete.")
sys.stdout.write("Model prediction... ")
sys.stdout.flush()

# predicted = model3.predict(X_prime)
# predicted = model3.predict(X)

predicted_results = [i.predict(X) for i in models]

print("complete.")

for predicted in predicted_results:
    print(metrics.classification_report(Y, predicted))
    print("Accuracy: ", metrics.accuracy_score(Y, predicted))
    # print(metrics.roc_curve(Y, predicted))

