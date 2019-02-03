# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 14:08:24 2019

@author: LubichJD
@credit: Kyle Thomas had some nice code that expanded the dictionary
@credit: Tons of scikit learn usage guides and documetation

Homework 1 
  # 1. Write a function to take a list or dictionary of clfs and hypers ie use logistic regression, each with 3 different sets of hyper parrameters for each
  # 2. expand to include larger number of classifiers and hyperparmater settings
  # 3. find some simple data
  # 4. generate matplotlib plots that will assist in identifying the optimal clf and parampters settings
  # 5. Please set up your code to be run and save the results to the directory that its executed from
  # 6. Investigate grid search function

"""

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score # other metrics?
from sklearn.metrics import make_scorer
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.model_selection import GridSearchCV  
import os
import datetime
import itertools as it
from sklearn import datasets
import matplotlib.pyplot as plt
import matplotlib as mpl 

currentDT = datetime.datetime.now()
output_dir = currentDT.strftime("%Y%m%d_%H%M%S")

if not os.path.exists(output_dir):
  os.makedirs(output_dir)

s = output_dir + '//summary.txt'  
d = output_dir + '//details.txt'  
    
n_folds = 10


iris = datasets.load_iris()
M = iris.data[:, :2]  # we only take the first two features.
L = iris.target

data = (M, L, n_folds)

testInput = [
    {'clf': [LogisticRegression], "name" : "LogisticRegression", "penalty" : ['l1', 'l2'], "tol": [0.001,0.01,0.1]},
    {'clf': [GaussianProcessClassifier], "name" : "GaussianProcessClassifier", "kernel": [1.0 * RBF(1.0)]},
    {'clf': [KNeighborsClassifier], "name" : "KNeighborsClassifier", "n_neighbors" : [3, 5, 7], "algorithm" : ['auto', 'ball_tree', 'kd_tree', 'brute']},
    #'SVC': [{"kernel" : ["linear"], "C" : [0.025]},
    #         {"gamma" : [2], "C" : [1]}],
    {'clf': [DecisionTreeClassifier], "name" : "DecisionTreeClassifier", "max_depth" : [3, 5, 7, 10], "min_samples_leaf" : [2, 3, 4, 5]},
    #'RandomForestClassifier': {"max_depth" : [5], "n_estimators" : [10], "max_features" : [1]},
    {'clf': [MLPClassifier], "name" : "MLPClassifier", "alpha" : [1], "learning_rate" : ['constant', 'invscaling', 'adaptive']},
    {'clf': [AdaBoostClassifier], "name" : "AdaBoostClassifier", "n_estimators" : [30, 50, 100, 200], "learning_rate" : [.01, .1, 1, 10]},
    #'GaussianNB': {},
    {'clf': [QuadraticDiscriminantAnalysis], "name" : "QuadraticDiscriminantAnalysis"},
    {'clf': [RandomForestClassifier], "name": "RandomForestClassifier", "min_samples_split": [2,3,4], "max_depth" : [3,5,8], "n_estimators" : [5, 10]}
 ]


def run(a_clf, data, clf_hyper={}):
  M, L, n_folds = data # unpack data containter
  kf = KFold(n_splits=n_folds) # Establish the cross validation
  ret = {} # classic explicaiton of results
 
  for ids, (train_index, test_index) in enumerate(kf.split(M, L)):
    clf = a_clf(**clf_hyper) # unpack paramters into clf is they exist
 
    clf.fit(M[train_index], L[train_index])
 
    pred = clf.predict(M[test_index])
 
    ret[ids]= {'clf': clf,
               'train_index': train_index,
               'test_index': test_index,
               'accuracy': accuracy_score(L[test_index], pred)}
  return ret
 

##This is freaking awesome because it puts everything back into a dictionary
##Courtesy of Kyle Thomas
def unpackHypers(kwargs):
  #borrowed from https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in it.product(*vals):
        yield dict(zip(keys, instance))

def DeathToGridSearch(searchList):
# input is a list of lists with the format [model, hyperParamterdict], ideally the hyperparameters will be 
# expanded to accept a list of inputs right now it is just a single value to make sure variable 
# unpacking is passed to the classifer correctly
  
  results = {}

  for searchVar in searchList:
    
    classifier_name = searchVar.pop('name', None)
    allHypers = list(unpackHypers(searchVar))
    print("Performing Grid Search on:", classifier_name, file=open(d,'a'))
    print("All Hyperparameters Raw:", allHypers, file=open(d,'a'))

    combination_results = []

    for search in allHypers:
          classifier = search.pop('clf', None) #if clf is a key pop it, return none otherwise
           #if clf is a key pop it, return none otherwise
          hypers = search
          print("Hyperparameter set:", hypers, file=open(d,'a'))
          
          try:
          # some classifiers might fail, try to run it and if failure continue
            res = run(classifier, data, hypers)
          except Exception as e:      
            res = "Classifier {} failed with the following error:\n{}\nResuming Search".format(str(clf), e)
          
          print("Results:", res, file=open(d,'a'))
          combination_results.append([hypers, res])
    
    results[classifier_name] = combination_results

  return results

results = DeathToGridSearch(testInput)

classifier_scores = []

##Parse Results into Class Name and List of Hyperparameter Options and Results
for classifier_name, classifier_results in results.items():
  
  ## For each Classifier create an output directory
  if not os.path.exists(output_dir + '\\' + classifier_name):
    os.makedirs(output_dir + '\\' + classifier_name)
    
  c = output_dir + '\\' + classifier_name + '\\hyperparameter_details.txt'
    
  trial_scores = []
    
  ## For each set of hyperparameters get the results
  for trial_id, trial in enumerate(classifier_results):
    hyperparameters = trial[0]
    trial_results = trial[1]
    
    print("**************************************************", file=open(c,'a'))
    print("Hyperparameter Set:", trial_id, file=open(c,'a'))
    print("Hyperparameters:", trial[0], file=open(c,'a'))
    
    ## For each set of hyperparameters create an output directory
    dir_name = output_dir + '\\' + classifier_name + '\\' + 'Hyperparameters' + str(trial_id)
    if not os.path.exists(dir_name):
      os.makedirs(dir_name)
    
    ## List out all of the accurcies from all folds in this trial result
    trial_accuracies = [fold['accuracy'] for fold_key, fold in trial_results.items()]
    print("Trial Accuracies:", trial_accuracies, file=open(c,'a'))

    trial_scores.append((sum(trial_accuracies)/len(trial_accuracies), trial_accuracies))
    print("Trial Average Accuracy by Fold:", sum(trial_accuracies)/len(trial_accuracies), file=open(c,'a'))
    print("Trial Std Accuracy:", np.std(trial_accuracies), file=open(c,'a'))

    ##Save an individual boxplot to the hyperparameter folder
    ######################################################################
    # Create a figure instance
    fig = plt.figure(1, figsize=(9, 6))
    
    # Create an axes instance
    ax = fig.add_subplot(111)
    
    # Create the boxplot
    ax.boxplot(trial_accuracies)
    fig.savefig(dir_name + '\\boxplot.png')
    plt.close(fig)
    ######################################################################
  
  ##Get the winning combination for a classifier  
  trial_scores_winner = trial_scores.index(max(trial_scores))
  trial_scores_winner_avg_value = max(trial_scores)[0]
  trial_scores_winner_values = max(trial_scores)[1]
  trial_scores_winner_parameters = str(classifier_results[trial_scores_winner][0])

  print("*******************************************************", file=open(c,'a'))
  print("******* Best Hyperparameters for ", classifier_name, " ************", file=open(c,'a'))
  print("trial_scores_winner", trial_scores_winner, file=open(c,'a'))
  print("trial_scores_winner_avg_value", trial_scores_winner_avg_value, file=open(c,'a'))
  print("trial_scores_winner_values", trial_scores_winner_values, file=open(c,'a'))
  print("trial_scores_winner_parameters", trial_scores_winner_parameters, file=open(c,'a'))
  
  classifier_scores.append((trial_scores_winner_avg_value, classifier_name, trial_scores_winner_values, trial_scores_winner_parameters))
  
  # Create a figure instance
  fig = plt.figure(1, figsize=(9, 6))
  
  # Create an axes instance
  ax = fig.add_subplot(111)
  
  # Create the boxplot
  ax.boxplot([scores for ave, scores in trial_scores])
  ax.set_title('Hyperparameter Trial Results')
  fig.savefig(output_dir + '\\' + classifier_name + '\\Hyperparameter_boxplot.png')
  plt.close(fig)
  
#############################################################
## Plot the best of the classifiers
#############################################################

classifier_scores_winner = classifier_scores.index(max(classifier_scores))
classifier_scores_winner_avg_value = max(classifier_scores)[0]
classifier_scores_winner_classifier = classifier_scores[classifier_scores_winner][1]
classifier_scores_winner_values = classifier_scores[classifier_scores_winner][2]
classifier_scores_winner_parameters = classifier_scores[classifier_scores_winner][3]

print("*****************************************************", file=open(s, 'a'))
print("************ Best Classifier ************************", file=open(s, 'a'))
print("*****************************************************", file=open(s, 'a'))
print("Best Overall Classifier:", classifier_scores_winner_classifier, file=open(s, 'a'))
print("Best Overall Classifier Avg Accuracy:", classifier_scores_winner_avg_value, file=open(s, 'a'))
print("Best Overall Classifier Accuracy Std:", np.std(classifier_scores_winner_values), file=open(s, 'a'))
print("Best Overall Classifier Parameters:", classifier_scores_winner_parameters, file=open(s, 'a'))
print("", file=open(s, 'a'))
  
classifier_names = []
accuracies = []
        
print("*****************************************************", file=open(s, 'a'))
print("************ Best of Classifiers Scores *************", file=open(s, 'a'))
print("*****************************************************", file=open(s, 'a'))
for classifier_results in classifier_scores:
  classifier_names.append(classifier_results[1])
  accuracies.append(classifier_results[2])
  
  print("*****************************************************", file=open(s, 'a'))
  print("Classifier Name:", classifier_results[1], file=open(s, 'a'))
  print("Classifier Average Accuracy:", classifier_results[0], file=open(s, 'a'))
  print("Classifier Accuracy Std:", np.std(classifier_results[2]), file=open(s, 'a'))
  print("Classifier Parameters:", classifier_results[3], file=open(s, 'a'))

  
# Create a figure instance
fig = plt.figure(1, figsize=(9, 8))
plt.tight_layout()
# Create an axes instance
ax = fig.add_subplot(111)

# Create the boxplot
ax.boxplot(accuracies)

## Custom x-axis labels
ax.set_xticklabels(classifier_names, rotation=65, fontsize=8)
ax.set_title('Best of the Classifiers')

fig.savefig(output_dir + '\\best_classifications_boxplot.png')
plt.close(fig)

print("******************* FINISHED **********************")
print("Best Overall Classifier:", classifier_scores_winner_classifier)
print("Best Overall Classifier Avg Accuracy:", classifier_scores_winner_avg_value)
print("Best Overall Classifier Accuracy Std:", np.std(classifier_scores_winner_values))
print("Best Overall Classifier Parameters:", classifier_scores_winner_parameters)
print("")
print("** Please view results in the directory:", output_dir)
print("")

  
  
  
  
  
  
  
  
  
  