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

from sklearn.metrics import matthews_corrcoef, make_scorer, roc_auc_score
import skflow
from sklearn import datasets, metrics, preprocessing

from sklearn.grid_search import GridSearchCV, RandomizedSearchCV

import  sys
from    tabulate    import  tabulate
###############################################################################
## Load Data Set
###############################################################################
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

X = training_dataset[:,2:52]
Y = training_dataset[:,1]


# Utility function to report best scores
def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
            score.mean_validation_score,
            np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")


###############################################################################
## Feature Selection
###############################################################################

X_new = SelectKBest(chi2, k=20).fit_transform(X, Y)

X = X_new
X = preprocessing.normalize(X)
print(X_new.shape)

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
              {'solver': ['svd', 'lsqr'],
                  # 'shrinkage': [None, 'auto', 0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 1],
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
    grid_search = RandomizedSearchCV(model, param_grid, n_iter=50, scoring=auc_scorer, cv=5)
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
        x = cross_validation.cross_val_score(model, X, Y, cv=5, scoring=mcc_scorer)
        y = cross_validation.cross_val_score(model, X, Y, cv=5, scoring=auc_scorer)

        results.append([name, x.mean(), y.mean()])
    print("Iteration: ", i )
    print(tabulate(results, headers=["Model", "MCC", "AUC"]))

###############################################################################
## TensorFlow
###############################################################################

# tensorflow_classifiers = [skflow.TensorFlowDNNClassifier(hidden_units=[10, 20, 10], n_classes=3), skflow.TensorFlowLinearClassifier(n_classes=3)]
# classifier_names = ["Deep Neural Network", "Linear Classifier"]

# for classifier, name in zip(tensorflow_classifiers, classifier_names):
#     classifier.fit(X, Y)
#     print("Working on %s" % name)
#     score = metrics.accuracy_score(classifier.predict(X), Y)
#     print("Accuracy: %f" % score)
