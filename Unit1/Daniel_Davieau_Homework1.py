# copied and adapted from Christopher Havenstein office hours presentations
import numpy as np
from itertools import product
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn import datasets
from IPython.display import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

x = np.array([[1,2],[3,4],[4,5],[4,5],[4,5],[4,5],[4,5],[4,5],[4,5],[4,5],[4,5],[4,5],[4,5],[4,5],[4,5],[4,5]])
y = np.random.choice([0,1], size=(x.shape[0],), p=[1./3, 2./3])
# iris = datasets.load_iris()
# x = iris.data
# y = iris.target
n_folds = 5
dataInputVariable = (x,y,n_folds)



def run(classifierInputVariable, dataInputVariable, parameterInputVariable={}):
    x, y, n_folds = dataInputVariable
    kf = KFold(n_splits=n_folds)
    ret = {}
    for a, (train_index, test_index) in enumerate(kf.split(x, y)):
        aClassifier = classifierInputVariable(**parameterInputVariable)
        aClassifier.fit(x[train_index], y[train_index])
        pred = aClassifier.predict(x[test_index])
        ret[a]= {'Classifier Function Used': aClassifier,
                 'train_index': train_index,
                 'test_index': test_index,
                 'accuracy': accuracy_score(y[test_index], pred)}
    return ret

# results={}
clfsAccuracyDict = {}
classifiersList = [RandomForestClassifier, LogisticRegression,SVC]
classifierParametersDictionary = {'RandomForestClassifier': {"min_samples_split": [2,3,4],"n_estimators":[100],"n_jobs":[2,4,6]},
                                  'LogisticRegression': {"tol": [0.001,0.01,0.1],"solver":['lbfgs'],"multi_class":['auto']},
                                 'SVC': {"C": [1.1, 0.5],"gamma":['scale']}}                        
for classifier in classifiersList:
    classifierString = str(classifier)
    for outerKey, outerValue in classifierParametersDictionary.items():
        if outerKey in classifierString:
            innerKey,innerValue = zip(*outerValue.items())
            for values in product(*innerValue):
                parameterInputVariable = dict(zip(innerKey, values))
                results = run(classifier, dataInputVariable, parameterInputVariable)
                for key in results:
                    k1 = results[key]['Classifier Function Used']
                    v1 = results[key]['accuracy']
                    k1Test = str(k1)
                    k1Test = k1Test.replace('            ',' ')
                    k1Test = k1Test.replace('          ',' ')
                    if k1Test in clfsAccuracyDict:
                        clfsAccuracyDict[k1Test].append(v1)
                    else:
                        clfsAccuracyDict[k1Test] = [v1] #create a new key (k1Test) in clfsAccuracyDict with a new value, (v1)
#                         print(clfsAccuracyDict)   


n = max(len(v1) for k1, v1 in clfsAccuracyDict.items())

# for naming the plots
filename_prefix = 'clf_Histograms_'

# initialize the plot_num counter for incrementing in the loop below
plot_num = 1 

# Adjust matplotlib subplots for easy terminal window viewing
left  = 0.125  # the left side of the subplots of the figure
right = 0.9    # the right side of the subplots of the figure
bottom = 0.1   # the bottom of the subplots of the figure
top = 0.6      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for space between subplots,
               # expressed as a fraction of the average axis width
hspace = 0.2   # the amount of height reserved for space between subplots,
               # expressed as a fraction of the average axis height

#create the histograms
for k1, v1 in clfsAccuracyDict.items():
    # for each key in our clfsAccuracyDict, create a new histogram with a given key's values 
    fig = plt.figure(figsize =(20,10)) # This dictates the size of our histograms
    ax  = fig.add_subplot(1, 1, 1) # As the ax subplot numbers increase here, the plot gets smaller
    plt.hist(v1, facecolor='green', alpha=0.75) # create the histogram with the values
    ax.set_title(k1, fontsize=30) # increase title fontsize for readability
    ax.set_xlabel('Classifer Accuracy (By K-Fold)', fontsize=25) # increase x-axis label fontsize for readability
    ax.set_ylabel('Frequency', fontsize=25) # increase y-axis label fontsize for readability
    ax.xaxis.set_ticks(np.arange(0, 1.1, 0.1)) # The accuracy can only be from 0 to 1 (e.g. 0 or 100%)
    ax.yaxis.set_ticks(np.arange(0, n+1, 1)) # n represents the number of k-folds
    ax.xaxis.set_tick_params(labelsize=20) # increase x-axis tick fontsize for readability
    ax.yaxis.set_tick_params(labelsize=20) # increase y-axis tick fontsize for readability
    #ax.grid(True) # you can turn this on for a grid, but I think it looks messy here.

    # pass in subplot adjustments from above.
    
    plt.subplots_adjust(left=left, right=right, bottom=bottom, top=top, wspace=wspace, hspace=hspace)
    plot_num_str = str(plot_num) #convert plot number to string
    filename = filename_prefix + plot_num_str # concatenate the filename prefix and the plot_num_str
    plt.savefig(filename, bbox_inches = 'tight') # save the plot to the user's working directory
    plot_num = plot_num+1 # increment the plot_num counter by 1
    
plt.show()