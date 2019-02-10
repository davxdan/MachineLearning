# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 17:31:47 2019

@author: Chris
"""

import functools
import io
import numpy as np
import sys
import numpy.lib.recfunctions as rfn
import time

#NumPy Cheatsheet - https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Numpy_Python_Cheat_Sheet.pdf


#to fix a bug in np.genfromtxt when Python Version (sys.version_info) is 3 or greater. 
# https://stackoverflow.com/questions/23319266/using-numpy-genfromtxt-gives-typeerror-cant-convert-bytes-object-to-str-impl


genfromtxt_old = np.genfromtxt
@functools.wraps(genfromtxt_old)
def genfromtxt_py3_fixed(f, encoding="utf-8", *args, **kwargs):
  if isinstance(f, io.TextIOBase):
    if hasattr(f, "buffer") and hasattr(f.buffer, "raw") and \
    isinstance(f.buffer.raw, io.FileIO):
      # Best case: get underlying FileIO stream (binary!) and use that
      fb = f.buffer.raw
      # Reset cursor on the underlying object to match that on wrapper
      fb.seek(f.tell())
      result = genfromtxt_old(fb, *args, **kwargs)
      # Reset cursor on wrapper to match that of the underlying object
      f.seek(fb.tell())
    else:
      # Not very good but works: Put entire contents into BytesIO object,
      # otherwise same ideas as above
      old_cursor_pos = f.tell()
      fb = io.BytesIO(bytes(f.read(), encoding=encoding))
      result = genfromtxt_old(fb, *args, **kwargs)
      f.seek(old_cursor_pos + fb.tell())
  else:
    result = genfromtxt_old(f, *args, **kwargs)
  return result

if sys.version_info >= (3,):
  np.genfromtxt = genfromtxt_py3_fixed

## HW notes:
'''    
A medical claim is denoted by a claim number ('Claim.Number'). Each claim consists of one or more medical lines denoted by a claim line number ('Claim.Line.Number').

1. J-codes are procedure codes that start with the letter 'J'.

     A. Find the number of claim lines that have J-codes.

     B. How much was paid for J-codes to providers for 'in network' claims?

     C. What are the top five J-codes based on the payment to providers?



2. For the following exercises, determine the number of providers that were paid for at least one J-code. Use the J-code claims for these providers to complete the following exercises.

    A. Create a scatter plot that displays the number of unpaid claims (lines where the ‘Provider.Payment.Amount’ field is equal to zero) for each provider versus the number of paid claims.

    B. What insights can you suggest from the graph?

    C. Based on the graph, is the behavior of any of the providers concerning? Explain.



3. Consider all claim lines with a J-code.

     A. What percentage of J-code claim lines were unpaid?

     B. Create a model to predict when a J-code is unpaid. Explain why you choose the modeling approach.

     C. How accurate is your model at predicting unpaid claims?

      D. What data attributes are predominately influencing the rate of non-payment?
'''


#Read the two first two lines of the file.
with open('data\claim.sample.csv', 'r') as f:
    print(f.readline())
    print(f.readline())

    
#Colunn names

names = ["V1","Claim.Number","Claim.Line.Number",
         "Member.ID","Provider.ID","Line.Of.Business.ID",
         "Revenue.Code","Service.Code","Place.Of.Service.Code",
         "Procedure.Code","Diagnosis.Code","Claim.Charge.Amount",
         "Denial.Reason.Code","Price.Index","In.Out.Of.Network",
         "Reference.Index","Pricing.Index","Capitation.Index",
         "Subscriber.Payment.Amount","Provider.Payment.Amount",
         "Group.Index","Subscriber.Index","Subgroup.Index",
         "Claim.Type","Claim.Subscriber.Type","Claim.Pre.Prince.Index",
         "Claim.Current.Status","Network.ID","Agreement.ID"]

#https://docs.scipy.org/doc/numpy-1.12.0/reference/arrays.dtypes.html
'''
typesCheck = [np.dtype(float), np.dtype(float), np.dtype(float), np.dtype(float),
         np.dtype(object), np.dtype(float), np.dtype(float), np.dtype(object),
         np.dtype(object), np.dtype(object), np.dtype(object), np.dtype(float),
         np.dtype(object), np.dtype(object), np.dtype(object), np.dtype(object),
         np.dtype(object), np.dtype(object), np.dtype(float), np.dtype(float),
         np.dtype(float), np.dtype(float), np.dtype(float), np.dtype(object),
         np.dtype(object), np.dtype(object), np.dtype(float), np.dtype(object),
         np.dtype(object)]
'''

#data types after using typesCheck instead of types in the below function
types = ['S8', 'f8', 'i4', 'i4', 'S14', 'S6', 'S6', 'S6', 'S4', 'S9', 'S7', 'f8',
         'S5', 'S3', 'S3', 'S3', 'S3', 'S3', 'f8', 'f8', 'i4', 'i4', 'i4', 'S3', 
         'S3', 'S3', 'S4', 'S14', 'S14']

#https://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.genfromtxt.html


#sys.setrecursionlimit(10000)

#from io import BytesIO


#inpstream = io.open('data\claim.sample.csv','r')
#creates array of structured arrays
CLAIMS = np.genfromtxt('data\claim.sample.csv', dtype=types, delimiter=',', names=True, 
                       usecols=[0,1,2,3,4,5,
                                6,7,8,9,10,11,
                                12,13,14,15,16,
                                17,18,19,20,21,
                                22,23,24,25,26,
                                27,28])

print(CLAIMS.dtype)

#Notice the shape differs since we're using structured arrays.
print(CLAIMS.shape)

#However, you can still slice into it to get a specific row.
print(CLAIMS[0])

#Slice into to get a specific value.
print(CLAIMS[0][1])

#Get the names
print(CLAIMS.dtype.names)

#Slice into a column
print(CLAIMS['MemberID'])

#Slice into a column and a row value
print(CLAIMS[0]['MemberID'])


#String Operations in NumPy - https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.char.html

#Sorting, Searching, and Counting in NumPy - https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.sort.html


# You might see issues here: https://stackoverflow.com/questions/23319266/using-numpy-genfromtxt-gives-typeerror-cant-convert-bytes-object-to-str-impl

test = 'J'
test = test.encode()

#Example of substring searching
np.core.defchararray.find(CLAIMS['ProcedureCode'],test)

#We only want the non -1 values
JcodeIndexes = np.flatnonzero(np.core.defchararray.find(CLAIMS['ProcedureCode'],test)!=-1)

print(JcodeIndexes)

#Using those indexes, subset CLAIMS to only Jcodes
Jcodes = CLAIMS[JcodeIndexes]

print(Jcodes)

print(Jcodes.dtype.names)

#QUESTION: How do you find the number of claim lines that have J-codes with "Jcodes"?
#You can figure this out. :)


#Sorted Jcodes, by ProviderPaymentAmount
Sorted_Jcodes = np.sort(Jcodes, order='ProviderPaymentAmount')


# Reverse the sorted Jcodes
Sorted_Jcodes = Sorted_Jcodes[::-1]
# [7, 6, 5, 4, 3, 2, 1]

# What are the top five J-codes based on the payment to providers?

# We still need to group the data
print(Sorted_Jcodes[:10])

# You can subset it...
ProviderPayments = Sorted_Jcodes['ProviderPaymentAmount']
Jcodes = Sorted_Jcodes['ProcedureCode']

#recall their data types
Jcodes.dtype
ProviderPayments.dtype

Jcodes[:3]
ProviderPayments[:3]



#Join arrays together
arrays = [Jcodes, ProviderPayments]

#https://www.numpy.org/devdocs/user/basics.rec.html
Jcodes_with_ProviderPayments = rfn.merge_arrays(arrays, flatten = True, usemask = False)

# What does the result look like?
print(Jcodes_with_ProviderPayments[:3])

Jcodes_with_ProviderPayments.shape




#http://esantorella.com/2016/06/16/groupby/
#A fast GroupBy class
class Groupby:
    def __init__(self, keys):
        _, self.keys_as_int = np.unique(keys, return_inverse = True)
        self.n_keys = max(self.keys_as_int)
        self.set_indices()
        
    def set_indices(self):
        self.indices = [[] for i in range(self.n_keys+1)]
        for i, k in enumerate(self.keys_as_int):
            self.indices[k].append(i)
        self.indices = [np.array(elt) for elt in self.indices]
        
    def apply(self, function, vector, broadcast):
        if broadcast:
            result = np.zeros(len(vector))
            for idx in self.indices:
                result[idx] = function(vector[idx])
        else:
            result = np.zeros(self.n_keys)
            for k, idx in enumerate(self.indices):
                result[self.keys_as_int[k]] = function(vector[idx])

        return result


#See how long the groupby takes
start = time.clock()
#grouped = Groupby(Jcodes)

#perform the groupby to get the group sums
group_sums = Groupby(Jcodes).apply(np.sum, ProviderPayments, broadcast=False)
print('time to compute group sums once with Grouped: {0}'\
      .format(round(time.clock() - start, 3)))


group_sums.shape

np.set_printoptions(threshold=500, suppress=True)
print(group_sums)

#How do we get the JCodes for the group sums?
#Look up at the class Groupby
unique_keys, indices = np.unique(Jcodes, return_inverse = True)

print(unique_keys)
print(indices)

len(unique_keys)
len(group_sums)    

print(group_sums)

#Zip it and sort it.
zipped = zip(unique_keys, group_sums)  # python 3
sorted_group_sums = sorted(zipped, key=lambda x: x[1])

print(sorted_group_sums)