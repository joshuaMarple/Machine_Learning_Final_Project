import  numpy                          as           np
import  urllib
from    sklearn.tree                   import       DecisionTreeClassifier
from    sklearn.neighbors              import       KNeighborsClassifier
from    sklearn.svm                    import       SVC
from    sklearn                        import       metrics
from    sklearn.ensemble               import       RandomForestClassifier
from    sklearn.naive_bayes            import       GaussianNB
from    sklearn.discriminant_analysis  import       LinearDiscriminantAnalysis  as   LDA
from    sklearn                        import       cross_validation

import  statistics
import  sys
from    tabulate    import  tabulate

sys.stdout.write("Beginning data set acquisition... ")
sys.stdout.flush()

# training_url      = "http://www.ittc.ku.edu/~jhuan/EECS738_F15/slides/EECS738_Train.csv"
# training_raw_data = urllib.urlopen(training_url)
training_raw_data   = open("EECS738_Train.csv", "r")
training_dataset    = np.loadtxt(training_raw_data, delimiter = ",")

# testing_url      = "http://www.ittc.ku.edu/~jhuan/EECS738_F15/slides/EECS738_Test.csv"
# testing_raw_data = urllib.urlopen(testing_url)
# testing_dataset  = np.loadtxt(testing_raw_data, delimiter=",")

print("complete.")
sys.stdout.write("Loading into datasets... ")
sys.stdout.flush()

X = training_dataset[:,2:10]
Y = training_dataset[:,1]

print("complete.")

sys.stdout.write("Model fitting... ")
sys.stdout.flush()

models = [KNeighborsClassifier(n_neighbors=3),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        LDA(),
        GaussianNB()]

model_names = ["K Neighbors", "Decison Tree", "Random Forest", "LDA", "Naive Bayes"]

print("complete.")
sys.stdout.write("Model prediction... ")
sys.stdout.flush()

print("complete.")

results = []
for model, name in zip(models, model_names):
    tmp_results = []
    x = cross_validation.cross_val_score(model, X, Y, cv=5)
    results.append([name, statistics.mean(x), statistics.stdev(x)])
print()
print(tabulate(results, headers=["Model", "Accuracy", "+/-"]))
