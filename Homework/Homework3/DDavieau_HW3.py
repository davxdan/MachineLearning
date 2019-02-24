#Snippets and assistance provided by Christopher Havenstein
import numpy as np
#%%
#Set a seed for consistent results
np.random.seed(seed=1)
##Generate 10 rows of 4 random values which each sum to 1 for peoples'
# preferences
npRatings=np.array(np.random.dirichlet(np.ones(4),size=10))
#Peoples' Preferenmces in nested dictionaries (Automate this if I have time)
people = {'Jane': {'willingness to travel': 0.1596993,
                  'desire for new experience':0.67131344,
                  'cost':0.15006726,
                  #'indian food':1,
                  #'Mexican food':1,
                  #'hipster points':3,
                  'vegetarian': 0.01892,
                  },
          'Bob': {'willingness to travel': 0.63124581,
                  'desire for new experience':0.20269888,
                  'cost':0.01354308,
                  #'indian food':1,
                  #'Mexican food':1,
                  #'hipster points':3,
                  'vegetarian': 0.15251223,
                  },
           'Charlie': {'willingness to travel': 0.312165,
                  'desire for new experience':0.472797,
                  'cost':0.0874337,
                  #'indian food':1,
                  #'Mexican food':1,
                  #'hipster points':3,
                  'vegetarian': 0.127604,
                  },
            'Daniel': {'willingness to travel': 0.24375,
                  'desire for new experience':0.521329,
                  'cost':0.0568194,
                  #'indian food':1,
                  #'Mexican food':1,
                  #'hipster points':3,
                  'vegetarian': 0.178102,
                  },
            'Emma': {'willingness to travel': 0.467763,
                  'desire for new experience':0.503437,
                  'cost':0.0198863,
                  #'indian food':1,
                  #'Mexican food':1,
                  #'hipster points':3,
                  'vegetarian': 0.00891349,
                  },
            'Felicia': {'willingness to travel': 0.0632823,
                  'desire for new experience':0.715663,
                  'cost':0.0351984,
                  #'indian food':1,
                  #'Mexican food':1,
                  #'hipster points':3,
                  'vegetarian': 0.185856,
                  },
            'Gary': {'willingness to travel': 0.577415,
                  'desire for new experience':0.138869,
                  'cost':0.214609,
                  #'indian food':1,
                  #'Mexican food':1,
                  #'hipster points':3,
                  'vegetarian': 0.0691065,
                  },
            'Helen': {'willingness to travel': 0.1596993,
                  'desire for new experience':0.67131344,
                  'cost':0.15006726,
                  #'indian food':1,
                  #'Mexican food':1,
                  #'hipster points':3,
                  'vegetarian': 0.01892,
                  },
            'Igor': {'willingness to travel': 0.1596993,
                  'desire for new experience':0.67131344,
                  'cost':0.15006726,
                  #'indian food':1,
                  #'Mexican food':1,
                  #'hipster points':3,
                  'vegetarian': 0.01892,
                  },
            'Jimmy': {'willingness to travel': 0.1596993,
                  'desire for new experience':0.67131344,
                  'cost':0.15006726,
                  #'indian food':1,
                  #'Mexican food':1,
                  #'hipster points':3,
                  'vegetarian': 0.01892,
                  },
                  }
#%%
###############################################################################
#Transform the user data into a matrix(M_people).
#Keep track of column and row ids.
#Normalize the points for each user -- make their preferences add to 1.
###############################################################################
peopleKeys, peopleValues = [], []
lastKey = 0
for k1, v1 in people.items():
    row = []
    for k2, v2 in v1.items():
        peopleKeys.append(k2)
        if k1 == lastKey:
            row.append(v2)      
            lastKey = k1
        else:
            peopleValues.append(row)
            row.append(v2)   
            lastKey = k1
M_people = np.array(peopleValues)
#M_people.shape
#np.sum(M_people, axis=1)
#%%
###############################################################################
# Transform the restaurant data into a matrix(M_restaurants)
# use the same column index.
###############################################################################
#%%
#Generate 10 rows of random values between 1 and 5
#Snipped and modded from Kthomas
names  = ['Flacos', 'Joes', 'McDonalds', 'BurgerKing', 'DannysTasteOfTexas ',
          'RedLobster', 'TGIFridays', 'OliveGarden', 'Derpburgers', 'PHO501']
cats = ['Distance', 'Novelty', 'Cost', 'Vegetarian']

def make_restaurants_dict(names, categories):
    restaurants = {}

    for n in names:
        npRandScores=np.random.randint(1,6,10)
        dict_input = dict(zip(categories, npRandScores))
        restaurants[n] = dict_input
    
    return restaurants

restaurants = make_restaurants_dict(names, cats)
#%%
restaurantsKeys, restaurantsValues = [], []
for k1, v1 in restaurants.items():
    for k2, v2 in v1.items():
        restaurantsKeys.append(k1+'_'+k2)
        restaurantsValues.append(v2)
#Noted that the shape is 8 but we need it to be the same shape as people matrix
#len(restaurantsValues)
#create np matrix and reshape to 2 by 4 in the same function
M_restaurants = np.reshape(restaurantsValues, (4,10))
#Verify shape
#M_restaurants.shape
#%%
























#%%
# Matrix multiplication

##########################Notes from Christopher Havenstein
## Dot products are the matrix multiplication of a row vectors and column vectors (n,p) * (p,n)
##  shape example: ( 2 X 4 ) * (4 X 2) = 2 * 2
## documentation: https://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html
## intuition: https://www.mathsisfun.com/algebra/matrix-multiplying.html
## What is a matrix product?
## https://en.wikipedia.org/wiki/Matrix_multiplication
## https://docs.scipy.org/doc/numpy/reference/generated/numpy.matmul.html#numpy.matmul
##However, this won't work...
###########################
#%%
#This errors because matrices aren't aligned properly. Inner dim must be same size
#np.matmul(restaurantsMatrix, peopleMatrix)
#restaurantsMatrix.shape, peopleMatrix.shape
#Out[10]: ((2, 4), (2, 4))
#%%

#Swap axis on peopleMatrix
newPeopleMatrix = np.swapaxes(peopleMatrix, 1, 0)
#%%
#https://docs.scipy.org/doc/numpy/reference/generated/numpy.swapaxes.html
newPeopleMatrix = np.swapaxes(peopleMatrix, 0, 1)
#%%
#restaurantsMatrix.shape, newPeopleMatrix.shape
#Out[15]: ((2, 4), (4, 2))
#%%
# The most imporant idea in this project is the idea of a linear combination.
# Informally describe what a linear combination is and how it will relate to our resturant matrix.

    #This is for you to answer! However....https://en.wikipedia.org/wiki/Linear_combination
    # essentially you are multiplying each term by a constant and summing the results.

# Choose a person and compute(using a linear combination) the top restaurant for them.  
# What does each entry in the resulting vector represent?

#Build intuition..
#Jane's score for Flacos
2*0.1596993 + 3*0.67131344 + 4*0.15006726 + 5*0.01892

#Bob's score for Flacos
2*0.63124581 + 3*0.20269888 + 4*0.01354308 + 5*0.15251223

#Jane's score for Joes
5*0.1596993 + 1*0.67131344 + 5*0.15006726 + 3*0.01892

#Bob's score for Joes
5*0.63124581 + 1*0.20269888 + 5*0.01354308 + 3*0.15251223

#%%
# Next compute a new matrix (M_usr_x_rest  i.e. an user by restaurant) from all people.  What does the a_ij matrix represent?
#Let's check our answers
results = np.matmul(restaurantsMatrix, newPeopleMatrix)
#%%


# Sum all columns in M_usr_x_rest to get optimal restaurant for all users.  What do the entry’s represent?
# I believe that this is what John and  is asking for, sum by columns
np.sum(results, axis=1)
#%%

# Now convert each row in the M_usr_x_rest into a ranking for each user and call it M_usr_x_rest_rank.   
# Do the same as above to generate the optimal resturant choice.
results

# Say that rank 1 is best

#reference: https://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html
# Argsort returns the indices that would sort an array - https://stackoverflow.com/questions/17901218/numpy-argsort-what-is-it-doing
# By default, argsort is in ascending order, but below, we make it in descending order and then add 1 since ranks start at 1
sortedResults = results.argsort()[::-1] +1
sortedResults

#What is the problem here? 
                               
