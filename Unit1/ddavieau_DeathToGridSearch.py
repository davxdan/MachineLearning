import numpy as np
from sklearn.metrics import accuracy_score # other metrics?
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

# adapt this code below to run your analysis

# Due before live class 2
# 1. Write a function to take a list or dictionary of clfs and hypers ie use logistic regression, each with 3 different sets of hyper parrameters for each

#Due before live class 3
# 2. expand to include larger number of classifiers and hyperparmater settings
# 3. find some simple data
# 4. generate matplotlib plots that will assist in identifying the optimal clf and parampters settings

#Due before live class 4
# 5. Please set up your code to be run and save the results to the directory that its executed from
# 6. Investigate grid search function

M = numpy.array([[1,2],[3,4],[4,5],[4,5],[4,5],[4,5],[4,5],[4,5]])
L = numpy.random.choice([0,1],size=(M.shape[0])) #creates an array of 1's in the same as rows in M
n_folds = 5
data = (M, L, n_folds) #creates a tuple; a tuple is same as list but immutable (Cant be changed)

def run(a_Classifier, data, Classifier_hyper={}): # {} are dictionary and [] are for list
  M, L, n_folds = data # unpack data containter
  kf = KFold(n_splits=n_folds) # Establish the cross validation
  Classifier_hyper= {'n_jobs': 2}
  ret = {} # classic explicaiton of results as dictionary
  for ids, (train_index, test_index) in enumerate(kf.split(M, L)):
        print("k fold = ", ids)
        print("            train indexes = ", train_index)
        print("            test indexes = ", test_index)
        Classifier = a_Classifier(**Classifier_hyper) # unpack paramters into clf is they exist
        Classifier.fit(M[train_index], L[train_index])
        pred = Classifier.predict(M[test_index])
        ret[ids]= {'Classifier': Classifier,
               'train_index': train_index,
               'test_index': test_index,
               'accuracy': accuracy_score(L[test_index], pred)},
  return ret

algorithmlist = [RandomForestClassifier,LogisticRegression]
for algorithms in algorithmlist:
    results = run(algorithms, data, Classifier_hyper={})
    print(results)