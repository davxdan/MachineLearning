# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 16:32:16 2019
@author: Jeremy Lubich
Machine Learning HW2
"""

import numpy as np
import numpy_groupies as npg
from tabulate import tabulate
from numpy.lib.recfunctions import append_fields
import matplotlib.pyplot as plt


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

types = ['S8', 'f8', 'i4', 'i4', 'S14', 'S6', 'S6', 'S6', 'S4', 'S9', 'S7', 'f8',
         'S5', 'S3', 'S3', 'S3', 'S3', 'S3', 'f8', 'f8', 'i4', 'i4', 'i4', 'S3', 
         'S3', 'S3', 'S4', 'S14', 'S14']




#creates array of structured arrays
claims_data = np.genfromtxt(
    'claim.sample.csv', 
    dtype=types, 
    delimiter=',', 
    names=True, 
    usecols=np.arange(0,29))

print('Length of all claims records')
len(claims_data)

test = 'J'
test = test.encode()

#Example of substring searching
np.core.defchararray.find(claims_data['ProcedureCode'],test)

#We want the ProcedureCodes which start with J 
JcodeIndexes = np.flatnonzero(np.core.defchararray.find(claims_data['ProcedureCode'],test)==1)

print('**********************************************')
print('Question 1 - A: 51029 claim lines with a J-Code')
print(len(JcodeIndexes))
print('**********************************************')

##########################################################################
# B: How much was paid for J-codes to providers for 'in network' claims?
##########################################################################

#Using those indexes, subset to only Jcodes
claims_data_jcodes = claims_data[JcodeIndexes]
len(claims_data_jcodes)

## Here are all of the column names for reference
#print(claims_data_jcodes.dtype.names)
#print(claims_data_jcodes.dtype)

'''
('V1', 'ClaimNumber', 'ClaimLineNumber', 'MemberID', 'ProviderID', 'LineOfBusinessID', 
'RevenueCode', 'ServiceCode', 'PlaceOfServiceCode', 'ProcedureCode', 'DiagnosisCode', 
'ClaimChargeAmount', 'DenialReasonCode', 'PriceIndex', 'InOutOfNetwork', 'ReferenceIndex', 
'PricingIndex', 'CapitationIndex', 'SubscriberPaymentAmount', 'ProviderPaymentAmount', 
'GroupIndex', 'SubscriberIndex', 'SubgroupIndex', 'ClaimType', 'ClaimSubscriberType', 
'ClaimPrePrinceIndex', 'ClaimCurrentStatus', 'NetworkID', 'AgreementID')
'''

## Valid values are I and O
np.unique(claims_data_jcodes['InOutOfNetwork'])

## Get all of the "In-network" claims
in_network_index = np.flatnonzero(np.core.defchararray.find(claims_data_jcodes['InOutOfNetwork'],'I'.encode())==1)

print('**********************************************')
print('Question 1 - B: $2,417,220.96 paid to in-network providers')
print(claims_data_jcodes[in_network_index]['ProviderPaymentAmount'].sum().round(2))
print('**********************************************')

##########################################################################
# C: What are the top five J-codes based on the payment to providers?
##########################################################################

#Create a dictionary of all the JCodes and their ID number
jcode_to_index_dict = {ni : indi for indi, ni in enumerate(set(claims_data_jcodes['ProcedureCode']))}
jcode_to_name_dict = {indi : ni for indi, ni in enumerate(set(claims_data_jcodes['ProcedureCode']))}

#Loop through all rows and lookup the jcode_index
jcode_ids = [jcode_to_index_dict[ni] for ni in claims_data_jcodes['ProcedureCode']]

## Make sure both dictionaries return the same stuff.
jcode_to_name_dict[1]
jcode_to_index_dict[b'"J2597"']

## Create 1 sum per procedure code
sum_per_jcode_id = npg.aggregate(jcode_ids, claims_data_jcodes['ProviderPaymentAmount'], func='sum')

## Zip together the jcode id, the original jcode name and the sum
zipped = zip(jcode_ids, jcode_to_name_dict.values(), sum_per_jcode_id) 

## Sort based on the sum amount descending
sorted_group_sums = sorted(zipped, key=lambda x: x[2], reverse=True)

print('**********************************************')
print('Question 1 - C: top five J-codes based on the payment to providers')
print(tabulate(sorted_group_sums[0:5], headers=['JCode_ID', 'JCode', 'SumProviderPaymentAmount']))
print('**********************************************')

##########################################################################
# 2. For the following exercises, determine the number of providers that 
#  were paid for at least one J-code. Use the J-code claims for these providers 
# to complete the following exercises.
##########################################################################

## Find all "paid" providers
paid_providers = np.unique(claims_data_jcodes[np.where(claims_data_jcodes['ProviderPaymentAmount'] > 0)]['ProviderID'])

## Get all of the claim lines for these "paid" providers
claims_data_jcodes_paid = claims_data_jcodes[np.where(np.isin(claims_data_jcodes['ProviderID'], paid_providers))]

## Create the isPaid columns
claims_data_jcodes_paid = append_fields(claims_data_jcodes_paid, 'isPaid', claims_data_jcodes_paid['ProviderPaymentAmount'] > 0)
claims_data_jcodes_paid = append_fields(claims_data_jcodes_paid, 'isPaidCount', (claims_data_jcodes_paid['ProviderPaymentAmount'] > 0) + 0)
claims_data_jcodes_paid = append_fields(claims_data_jcodes_paid, 'isNotPaidCount', (claims_data_jcodes_paid['ProviderPaymentAmount'] == 0) + 0)

## There were 51,029 claim lines for JCodes
len(claims_data_jcodes)
## There are 51,015 claim lines for providers with one or more paid JCodes
len(claims_data_jcodes_paid)

## There are 44,947 unpaid claims and 6,068 paid ones
isPaidNames, isPaidValues = np.unique(claims_data_jcodes_paid['isPaid'], return_counts=True)

print('**********************************************')
print('Question 2: How many claims are there for paid providers...')
print(len(claims_data_jcodes_paid))
print('Question 2: How many paid claims are there? isPaid = ...')
print(tabulate([isPaidValues], headers=isPaidNames.data))
print('**********************************************')

##########################################################################
# A. Create a scatter plot that displays the number of unpaid claims (lines 
#     where the ‘Provider.Payment.Amount’ field is equal to zero) for each provider 
#     versus the number of paid claims.
##########################################################################

#Create a dictionary of all the Providers and their ID number
provider_to_index_dict = {ni : indi for indi, ni in enumerate(set(claims_data_jcodes_paid['ProviderID']))}

#Loop through all rows and lookup the jcode_index so we can group by ids
provider_ids = [provider_to_index_dict[ni] for ni in claims_data_jcodes_paid['ProviderID']]

# Group by provider ids getting paid and not paid counts
sum_per_provider_paid = npg.aggregate(provider_ids, claims_data_jcodes_paid['isPaidCount'], func='sum')
sum_per_provider_not_paid = npg.aggregate(provider_ids, claims_data_jcodes_paid['isNotPaidCount'], func='sum')

## Zip together the original provider name and the sums
provider_sums = zip(provider_to_index_dict.keys(), sum_per_provider_paid, sum_per_provider_not_paid, (sum_per_provider_paid + sum_per_provider_not_paid), sum_per_provider_paid / (sum_per_provider_paid + sum_per_provider_not_paid)) 

## Sort based on the sum amount descending
provider_sums = sorted(provider_sums, key=lambda x: x[3], reverse=True)


#####################################################
## Create Scatterplot
fig, ax = plt.subplots()
fig.set_size_inches(12.5, 5.5)

# Plot each provider value
for provider in provider_sums:
  ax.scatter(provider[1], provider[2], label='Provider:' + provider[0].decode(), edgecolors='none')

# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

ax.grid(True)

# Add a slope line??
plt.plot([0, 1775],[0,14000])

plt.title('Scatterplot 1 - Claims Paid by Provider')
plt.xlabel('Paid Claims')
plt.ylabel('Not Paid Claims')
plt.show()
#####################################################

### Sort based on the ratio amount descending
#provider_paid_ratio = sorted(zip([x[0].decode() for x in provider_sums], [x[1] / x[3] for x in provider_sums]), key=lambda x: x[1], reverse=True)
#providers_sorted = [x[0] for x in provider_paid_ratio]
#providers_paid_ratio_sorted = [x[1] for x in provider_paid_ratio]

######################################################
### Create Barchart
#fig, ax = plt.subplots()
#fig.set_size_inches(12.5, 5.5)
#
#ticklabels = []
#
## Plot each provider value
#for i, provider in enumerate(provider_paid_ratio):
#  ticklabels.append(provider[0])
#  ax.bar(i, provider[1], label='Provider:' + provider[0])
#
## Shrink current axis by 20%
#box = ax.get_position()
#ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
#
## Put a legend to the right of the current axis
#ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#
#ax.grid(True)
#ax.set_xticks(np.arange(len(ticklabels)))
#ax.set_xticklabels(ticklabels, rotation=90)
#
#
#plt.title('Percent of Claims Paid by Provider')
#plt.xlabel('Providers')
#plt.ylabel('Percent of Paid Claims')
#plt.show()
######################################################

print('**********************************************')
print('Question 2 - B. What insights can you suggest from the graph?')
print('We see there are a mixture of types of providers who pay payments. Some providers have a very small amount of claims and tend to pay them, while many other providers have a large number of claims and seem to pay a small percentage of them.')
print('I\ve added a line which bisects the top performers from the bottom performers. In this case, its those on the lower side of the line which are paying in a greater percentage of claims than not based on a comparison of their peers.')
print('**********************************************')

#####################################################
## Create 2nd Scatterplot
fig, ax = plt.subplots()
fig.set_size_inches(12.5, 5.5)

# Plot each provider value
for provider in provider_sums:
  ax.scatter(provider[3], provider[4], label='Provider:' + provider[0].decode(), edgecolors='none')
  ax.text(provider[3], provider[4], provider[0].decode(), fontsize=9)
  
  
# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

ax.grid(True)

plt.title('Scatterplot 2 - Percent Claims Paid by Provider')
plt.xlabel('Total Claims Submitted')
plt.ylabel('Percent Claims Paid')
plt.show()
#####################################################

print('**********************************************')
print('Question 2 - C. Based on the graph, is the behavior of any of the providers concerning? Explain.')
print('I\'ve created an additional scatterplot to help us analyze the performance of the providers which are not providing a good rate of claim payments.')
print('Along the x axis you see the total number of claims submitted and the percentage of claims paid on the y axis.')
print('The scatterplot naturally groups providers into 3 clusters. ')
print('Cluster 1 on the lower left are the providers with a relatively low amount of claims and a low percentage of paying them.')
print('Cluster 2 on the upper left are the providers with a relatively low amount of claims and a high percentage of paying them.')
print('Cluster 3 on the lower right are the providers with a relatively high amount of claims and a high percentage of paying them.')
print('The most concerning provider is "FA0001387001" who has nearly 9k claims, but has paid nearly none of them. This provider is outlier amount the 4 big providers (Cluster 3) who have the most claims.')
print('**********************************************')

#3. Consider all claim lines with a J-code.
#     A. What percentage of J-code claim lines were unpaid?
#     B. Create a model to predict when a J-code is unpaid. Explain why you choose the modeling approach.
#     C. How accurate is your model at predicting unpaid claims?
#     D. What data attributes are predominately influencing the rate of non-payment?
     