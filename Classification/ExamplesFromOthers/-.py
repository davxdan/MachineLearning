##### runThemAll.py
# Credit: https://www.digitalocean.com/community/tutorials/how-to-build-a-machine-learning-classifier-in-python-with-scikit-learn
from Classification import Classification
from sklearn.datasets import load_breast_cancer, load_wine

if __name__ == "__main__":
    data = load_breast_cancer()
    shell = Classification(data, kfolds = 2)
    shell.run(verbose = True)
    shell.printBestScores()

#### Classification.py
# Libraries
import itertools as it
import numpy as np
import time
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from matplotlib import pyplot as plt


# Classes
from Skeletor import Skeletor
from BestScore import BestScore

# Decorator
# Credit: https://medium.com/pythonhive/python-decorator-to-measure-the-execution-time-of-methods-fa04cb6bb36d
def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts))
        else:
            print(f"Method [{method.__name__}] took {(te - ts):.2f} seconds")
        return result

    return timed

class Classification(Skeletor):
    defaultHypers = {}
    knnHypers = {}
    svcHypers = {}
    rfcHypers = {}
    lrHypers = {}
    bestScore = None

    def __init__(self, data, kfolds):
        Skeletor.__init__(self, data, kfolds)
        self.bestScore = BestScore()
        self.initialize()

    def initialize(self):
        self.initClassifiers()
        self.setDefaultHypers()
        Skeletor.initialize(self)

    def initClassifiers(self):
        self.clfHolder['KNN'] = KNeighborsClassifier
        self.clfHolder['SVC'] = SVC
        self.clfHolder['RFC'] = RandomForestClassifier
        self.clfHolder['LR'] = LogisticRegression

    def setDefaultHypers(self):
        self.defaultHypers = {
            'KNN': {'n_neighbors': np.linspace(1, 20, num = 20, dtype=int),
                    'weights': ['uniform','distance'],
                    'leaf_size': np.linspace(5,20, num = 10, dtype=int),
                    'metric': ['minkowski','euclidean']
                    },
            'SVC': {'C': np.linspace(.1, 1, num = 10),
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale'],
                    'tol': [1e-2, 1e-3, 1e-4]
                    },
            'LR':  {'C': np.linspace(.1, 1, num = 10),
                    'solver': ['liblinear'],
                    'max_iter': [1000],
                    'tol' : [1e-2, 1e-3, 1e-4]
                    },
            'RFC': {'n_estimators': [10,20,30],
                    'min_samples_leaf': [1, 2, 3],
                    'min_samples_split': [2 ,3, 4]
                    }
            }

    def setKnnHypers(self, hypers):
        self.knnHypers = hypers

    def setSvcHypers(self, hypers):
        self.svcHypers = hypers

    def setRfcHypers(self, hypers):
        self.rfcHypers = hypers

    def setLrHypers(self, hypers):
        self.lrHypers = hypers

    @timeit
    def run(self, verbose = False):
        for clfName in self.clfHolder.keys():
            if verbose:
                print(f"Running Classifier: {clfName}")

            clfType = self.clfHolder[clfName]
            kf = KFold(n_splits=self.kfolds)
            hyperPermutations = list(self.getHyperPermutations(clfName))

            for idx, (trainIdx, testIdx) in enumerate(kf.split(self.features, self.target)):
                if verbose:
                    print(f"Index Count: {idx}")

                for permutation in hyperPermutations:
                    accScore = None
                    precisionScore = None
                    recallScore = None
                    f1Score = None

                    clf = clfType(**permutation)

                    try:
                        clf.fit(self.features[trainIdx], self.target[trainIdx])
                        prediction = clf.predict(self.features[testIdx])
                        accScore = accuracy_score(self.target[testIdx], prediction)
                        precisionScore = precision_score(self.target[testIdx], prediction)
                        recallScore = recall_score(self.target[testIdx], prediction)
                        f1Score = f1_score(self.target[testIdx], prediction)
                    except Exception as e:
                        print(f"Classifier {clf} failed with the following error:\n{e}\nResuming Search.")

                    tmpResults = [idx, clfType, clf, accScore, precisionScore, recallScore, f1Score]

                    if self.results[f'{clfName}'] is None:
                        self.results[f'{clfName}'] = tmpResults
                    else:
                        self.results[f'{clfName}'].append(tmpResults)

    def getBestScores(self):
        if not self.bestScore.haveData():
            self.bestScore.setAllResults(self.results)

        self.bestScore.getBestScores()

    def getBestScore(self, metric):
        if not self.bestScore.haveData():
            self.bestScore.setAllResults(self.results)

        self.bestScore.getBestScore(metric)

    def printBestScores(self):
        if not self.bestScore.haveData():
            self.bestScore.setAllResults(self.results)

        self.bestScore.printBestScores()

    def printBestScore(self, metric):
        if not self.bestScore.haveData():
            self.bestScore.setAllResults(self.results)

        self.bestScore.printBestScore(metric)

    def getHyperPermutations(self, clfName):
        tmpHyper = self.defaultHypers[clfName]

        # Credit: https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists
        keys = tmpHyper.keys()
        vals = tmpHyper.values()
        for instance in it.product(*vals):
            yield dict(zip(keys, instance))

    def plotAccuracyResults(self, clf = None):
        tmpResults = {}
        for clfName in self.results:
            tmpResult = list()
            for tmp in self.results[clfName]:
                tmpResult.append(tmp[2])

            tmpResults[clfName] = tmpResult

        totalCols = len(tmpResults)

        fig = plt.figure()

        for colIdx, key in enumerate(tmpResults):
            plt.subplot(1, totalCols, colIdx + 1)
            plt.boxplot(tmpResults[key])
            plt.xlabel(key)
            plt.ylabel("Accuracy")

        fig.suptitle("Classifier Accuracy Report")
        plt.tight_layout(pad=3)
        plt.show()
		
##### Skeletor.py
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
from collections import defaultdict

class Skeletor(object):
    results = {}
    clfHolder = {}
    bestResults = {}

    metrics = ['accuracy', 'precision', 'recall', 'f1score']
    kfolds = None
    data = None
    features = None
    target = None
    featureNames = None
    targetNames = None
    trainSet = None
    testSet = None
    trainSetLabels = None
    testSetLabels = None


    def __init__(self, data, kfolds = 3):
        self.results = defaultdict(list)
        self.clfHolder = {}
        self.data = data
        self.kfolds = kfolds

    def initialize(self):
        self.features = self.data['data']
        self.featureNames = self.data['feature_names']
        self.target = self.data['target']
        self.targetNames = self.data['target_names']

        for metric in self.metrics:
            self.bestResults[metric] = -1

        for name in self.clfHolder.keys():
            self.results[f'{name}'] = []

    def run(self):
        for clfName in self.clfHolder.keys():
            clf = self.clfHolder[clfName]
            kf = KFold(n_splits=self.kfolds)

            for idx, (trainIdx, testIdx) in enumerate(kf.split(self.features, self.target)):
                clf.fit(self.features[trainIdx], self.target[trainIdx])
                prediction = clf.predict(self.features[testIdx])
                accScore = accuracy_score(self.target[testIdx], prediction)

                if self.results[f'{clfName}'] is None:
                    self.results[f'{clfName}'] = ([idx, clf, accScore])
                else:
                    self.results[f'{clfName}'].append([idx, clf, accScore])

    def plotResults(self, clf = None):
        allResults = {}
        for clfName in self.results:
            tmpResult = list()
            for tmp in self.results[clfName]:
                tmpResult.append(tmp[2])

            allResults[clfName] = tmpResult

        totalCols = len(allResults)

        fig = plt.figure()

        for colIdx, key in enumerate(allResults):
            plt.subplot(1, totalCols, colIdx + 1)
            plt.boxplot(allResults[key])
            plt.xlabel(key)
            plt.ylabel("Accuracy")

        fig.suptitle("Classifier Accuracy Report")
        plt.tight_layout(pad=3)
        plt.show()

#### Best Score.py
import numpy as np

class BestScore:
    bestScore = {}
    scoreLocation = {}
    allResults = None
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    haveScores = False

    def __init__(self):
        self.initialize()

    def initialize(self):
        tmpVal = 3
        for metric in self.metrics:
            self.scoreLocation[metric] = tmpVal
            tmpVal += 1

    def haveData(self):
        if self.allResults is None:
            return False
        return True

    def setAllResults(self, allResults):
        self.allResults = allResults

    def getBestScores(self):
        for metric in self.metrics:
            self.getBestScore(metric)

    def getBestScore(self, metric):
        tmpMax = []
        tmpResult = []
        for name in self.allResults.keys():
            result = max(self.allResults[name], key=lambda x: x[self.scoreLocation[metric]])
            tmpMax.append(result[self.scoreLocation[metric]])
            tmpResult.append(result)

        idx = np.argmax(tmpMax)
        self.bestScore[metric] = tmpResult[idx]

    def printBestScores(self):
        for metric in self.metrics:
            self.printBestScore(metric)

    def printBestScore(self, metric):
        if metric not in self.bestScore.keys():
            self.getBestScore(metric)

        print(f"Best {metric}: {self.bestScore[metric][self.scoreLocation[metric]]}")
        print(f"Model: {self.bestScore[metric][2]}\n\n")