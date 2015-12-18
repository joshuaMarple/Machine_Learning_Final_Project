import  numpy                          as      np
import  urllib
from ipdb import set_trace as st
from    sklearn.tree                   import  DecisionTreeClassifier
from    sklearn.neighbors              import  KNeighborsClassifier
from    sklearn.svm                    import  SVC
from    sklearn                        import  metrics
from    sklearn.ensemble               import  RandomForestClassifier
from    sklearn.naive_bayes            import  GaussianNB
from    sklearn.discriminant_analysis  import  LinearDiscriminantAnalysis  as             LDA
from    sklearn                        import  cross_validation,           preprocessing
from    sklearn.pipeline               import  make_pipeline
from    sklearn.preprocessing          import  PolynomialFeatures
from    sklearn.feature_selection      import  SelectKBest
from    sklearn.feature_selection      import  chi2

from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import matthews_corrcoef, make_scorer, roc_auc_score
from sklearn import datasets, metrics, preprocessing

from itertools import compress

from sklearn.grid_search import GridSearchCV, RandomizedSearchCV

import csv
import  sys
from    tabulate    import  tabulate

###############################################################################
## Load Data Set
###############################################################################

sys.stdout.write("Beginning data set acquisition... ")
sys.stdout.flush()

csvfile = open('prediction.txt', 'wt')
writer = csv.writer(csvfile, delimiter=",", quotechar='|', quoting=csv.QUOTE_MINIMAL)

training_raw_data   = open("EECS738_Train.csv", "r")
training_dataset    = np.loadtxt(training_raw_data, delimiter = ",")

test_raw_data = open("EECS738_Test.csv", "r")
testing_dataset = np.loadtxt(test_raw_data, delimiter = ",")

print("complete.")
sys.stdout.write("Loading into datasets... ")
sys.stdout.flush()

X = training_dataset[:,2:52]
Y = training_dataset[:,1]

###############################################################################
## Feature Selection
###############################################################################

best = SelectKBest(chi2, k=20)
X = best.fit_transform(X, Y)
filter = best.get_support()
new_test = []
for i, x in zip(filter, testing_dataset.T):
    if i:
        new_test.append(list(x))

testing_dataset = np.asarray(new_test).T

testing_dataset = preprocessing.normalize(testing_dataset)

X = preprocessing.normalize(X)

print("complete.")

sys.stdout.write("Model fitting... ")
sys.stdout.flush()

###############################################################################
## Define Models
###############################################################################

models = [KNeighborsClassifier(n_neighbors=150, n_jobs = -1),
        RandomForestClassifier(n_jobs = -1, n_estimators = 20),
        LDA()]

model_names = ["K Neighbors", "Random Forest", "LDA"]
param_grids = [{'n_neighbors': [1, 5, 10, 50, 200, 1000],
                  'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute'],
                  'p' : [1, 2, 3, 4, 5]},
              {'n_estimators': [1, 5, 10, 20, 50, 200],
                  'criterion': ['gini', 'entropy'],
                  'max_features': ['auto', 'sqrt', 'log2']},
              {'solver': ['lsqr', 'eigen'],
                  'shrinkage': [None, 'auto', 0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 1],
                  'n_components': list(range(1, 20, 5))}]

print("complete.")
sys.stdout.write("Model prediction... ")
sys.stdout.flush()

print("complete.")

###############################################################################
## Define Scorers
###############################################################################

mcc_scorer = make_scorer(matthews_corrcoef)
auc_scorer = make_scorer(roc_auc_score)

###############################################################################
## Randomized Search
###############################################################################

sys.stdout.write("Grid Search... ")
sys.stdout.flush()
new_models = []
for model, name, param_grid in zip(models, model_names, param_grids):
    grid_search = RandomizedSearchCV(model, param_grid, n_iter=20, scoring=auc_scorer, cv=5)
    grid_search.fit(X,Y)
    print(grid_search.best_estimator_)
    new_models.append(grid_search.best_estimator_)
    print(grid_search.grid_scores_)
print("complete.")
models = new_models

###############################################################################
## Run Models
###############################################################################

for i in range(1, 20):
    results = []
    for model, name, param_grid in zip(models, model_names, param_grids):
        tmp_results = []
        auc_sum = 0
        mcc_sum = 0
        folds = 5
        kf = StratifiedKFold(Y, folds)
        for train_index, test_index in kf:
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]
            model.fit(X_train, y_train)
            pred_class = model.predict(X_test)
            pred_test = model.predict_proba(X_test)[:, 1]

            auc_sum += metrics.roc_auc_score(y_test, pred_test)
            mcc_sum += metrics.matthews_corrcoef(y_test, pred_class)

        results.append([name, mcc_sum/folds, auc_sum/folds])
    print("Iteration: " + str( i ))
    print(tabulate(results, headers=["Model", "MCC", "AUC"]))

###############################################################################
## Predict Results
###############################################################################

models[2].fit(X, Y)
vals = models[2].predict(testing_dataset)
vals = vals.tolist()


for i in vals:
    writer.writerow([int(i)])


