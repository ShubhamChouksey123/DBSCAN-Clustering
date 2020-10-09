#!/usr/bin/env python
# coding: utf-8

# In[219]:


import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[220]:


# function to normalize distance matrix
def normalize(X):
    minn = float('inf')
    maxx = 0
    for i in range(len(X)):
        j = i + 1
        while j < len(X[0]):
            minn = min(minn, X[i][j])
            maxx = max(maxx, X[i][j])
            j += 1
    rng = maxx + minn
    for i in range(len(X)):
        j = i + 1
        while j < len(X[0]):
            X[i][j] = 1 - ((maxx - X[i][j]) / rng)
            X[j][i] = X[i][j]
            j += 1
    return X


# In[221]:


# function to return distance between datapoint1(dt1) and dapapoint2(dt2) 
def distance(dt1, dt2):
    return math.sqrt(np.sum((dt1 - dt2) ** 2))


# In[222]:


def fillDistanceMatrix(distanceMatrix, X):
    for i in range(len(X)):
        j = i
        while j < len(X):
            distanceMatrix[i][j] = distance(X[i], X[j])
            distanceMatrix[j][i] = distanceMatrix[i][j]
            j += 1


# In[223]:


''' function that finds core points 
by counting the number of points in the epsilon radius around that point and if count of point is greater than 
Minimum number of points it is a core point else not
'''  
def findCorePoints(X, corePoints, eps, MinPoints):
    for i in range(len(X)):
        count = 0
        for j in range(len(X)):
            if(distanceMatrix[i][j] <= eps):
                count += 1
        if(count >= MinPoints):
            corePoints.append(i)
    return corePoints


# In[224]:


'''Function that label the data Points
First we label each data point as -1(unlabbeled)
Then we run a loop for all the core points , if is is unlabelled then we increase the current_cluster_label variable
and start a new cluster and check for all points in the epsilon radius around that core point and label them as 
current_cluster_label 
'''
def Label_the_Clusters(corePoints, X, distanceMatrix, eps):
    label = [-1]*len(X)
    current_cluster_label = -1
    for i in range(len(corePoints)):
        if(label[i] == -1):
            current_cluster_label += 1
            label[i] = current_cluster_label
        for j in range(len(X)):
            if(i != j and distanceMatrix[i][j] <= eps and label[j] == -1):
                label[j] = current_cluster_label
    return label


# In[225]:


if __name__ == "__main__" : 

# reading the data from the CSV file 
    df = pd.read_csv("cancer.csv", usecols = ["radius_mean", "texture_mean",  "perimeter_mean",	"area_mean",	
        "smoothness_mean",	"compactness_mean",	"concavity_mean",	"concave points_mean",	"symmetry_mean",	
        "fractal_dimension_mean",	"radius_se",	"texture_se",	"perimeter_se",	"area_se",	"smoothness_se",	
        "compactness_se",	"concavity_se",	"concave points_se",	"symmetry_se",	"fractal_dimension_se",	
        "radius_worst",	"texture_worst",	"perimeter_worst",	"area_worst",	"smoothness_worst",	"compactness_worst",	
        "concavity_worst",	"concave points_worst",	"symmetry_worst",	"fractal_dimension_worst" ])

    X = df.iloc[:, ].values
    
    X = X.astype(float)
    
    k  = 2 
    X = np.array(X)
    X = normalize(X)

# Distance matrix where distanceMatrix[i][j] represent distance between the ith and jth datapoint 
    distanceMatrix = np.zeros((len(X), len(X)))
    fillDistanceMatrix(distanceMatrix, X)
    
    eps = 0.5
    MinPoints = 6
    eps = float(eps)
    MinPoints = float(MinPoints)
    
# finding core points in the dataset
    corePoints = []
    corePoints = findCorePoints(X, corePoints, eps, MinPoints)
    print("corePoints = ", corePoints)
    
# we label the points 
    label = Label_the_Clusters(corePoints, X, distanceMatrix, eps)
    
# if its not a core point nor the Border point then its a noise 
    for i in range(len(X)):
        if(label[i] == -1):
            label[i] = 6
            
# array coinsisting of abbrevations of colors      
    colors = ["g.", "r. ", "c." , "b.", "k.", "m.", "y."]
    
# plotting the Point and it is possible that number of clusters is greater than 6 so we are using modulo operator    
    for i in range(len(X)):
        plt.plot(X[i][0], X[i][1], colors[label[i]%6], markersize = 7)
    
    plt.xlabel("Radius_Mean")
    plt.ylabel("Texture_Mean")
    plt.show()

   
    


# In[ ]:




