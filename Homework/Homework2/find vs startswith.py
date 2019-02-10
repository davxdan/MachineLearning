# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 21:58:05 2019

@author: Chris
"""

import numpy as np

#A test string
test = 'J'
test = test.encode()

#A test NumPy array of type string
testStrArray = np.array(['Ja','JA', 'naJ', 'na' ],dtype='S9')

#Showing what the original string array looks like
print('Original String Array: ', testStrArray)

Test1Indexes = np.core.defchararray.startswith(testStrArray, test, start=0, end=None)

testResult1 = testStrArray[Test1Indexes]

#Showing what the original subset string array looks like with startswith()
print('Subset String Array with startswith(): ', testResult1)



TestIndexes = np.flatnonzero(np.core.defchararray.find(testStrArray,test)!=-1)

testResult2 = testStrArray[TestIndexes]

#Showing what the original subset string array looks like with find()
print('Subset String Array with find(): ', testResult2)