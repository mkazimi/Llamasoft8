#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns; sns.set()
import csv
from math import radians, cos, sin, asin, sqrt


def haversine(loc1, loc2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    
    lat1 = loc1[0]
    lon1 = loc1[1]
    lat2 = loc2[0]
    lon2 = loc2[1]
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

#returns a dictionary, which holds the stores for each cluster
def getStoresInCluster(customerInfo):
    clusterToStores = {}
    for i, row in customerInfo.iterrows():
        curCluster =row['cluster_label']
        if(curCluster not in clusterToStores):
            clusterToStores[curCluster] = set()
        clusterToStores[curCluster].add(row["store assignent"])
    return clusterToStores


def addToDict(dict, row):
    dict[row[0]] = [row[1], row[2]]

def nearestNeighbor(stores, start, end):
    prevStoreLoc = start
    storesNotVisited = {}
    totalDistance = 0
    #line below constructs the set of the stores not visited yet in the algorithm
    stores.apply(lambda store: addToDict(storesNotVisited, store), axis = 1)
    #while loop ends when every store has been visited
    while(len(storesNotVisited)!=0):
        curMin = float('inf')
        nextStore = -1
        #find the next store with the minimum distance from the previous
        for store in storesNotVisited:
            storeLoc = storesNotVisited[store]
            curDist = haversine(prevStoreLoc, storeLoc)
            if(curDist < curMin):
                curMin = curDist
                nextStore = store
        #adding to the totaldistance of the path
        totalDistance += curMin
        prevStoreLoc = storesNotVisited[nextStore]
        #remove nextStore from the set of storesNotVisited
        storesNotVisited.pop(nextStore)
    #adding the final part of the path from the final store to the center of the cluster
    totalDistance += haversine(prevStoreLoc, end)
    return totalDistance


df_demand = pd.read_csv('smalldemand.csv')
df_location = pd.read_csv('customerlocation.csv')
X = pd.merge(df_demand, df_location, left_on = 'customerID', right_on = 'customerID', how = 'left')

#IMPORTANT: THIS ASSUMES LAT AND LONG ARE COLUMNS 4 and 5
kmeans = KMeans(n_clusters = 3, init ='k-means++')
kmeans.fit(X[X.columns[4:6]])
X['cluster_label'] = kmeans.fit_predict(X[X.columns[4:6]])
centers = kmeans.cluster_centers_ # Coordinates of cluster centers
labels = kmeans.predict(X[X.columns[4:6]]) # Labels of each point
X.plot.scatter(x = 'lat', y = 'long', c=labels, s=50, cmap='viridis')

drivers = pd.read_csv('DriverStartLocation.csv')
stores = pd.read_csv('storeloc.csv')
#making sure driverId is of type string
drivers['DriverID'] = drivers.DriverID.astype(str)
#calculate the minimum path for each driver using the nearest neighbor algorithm above

with open('driverClusterMinDistances.csv', mode='w') as csv_file:
    fieldNames = ['cluster', 'driver', 'distance']
    writer = csv.writer(csv_file, delimiter=',', quotechar='"')
    writer.writerow(fieldNames)
   
    clusterToStores = getStoresInCluster(X)
    #looping through all the clusters
    for i, center in enumerate(centers):
        storesSubset = stores[stores["Store Id"].isin(clusterToStores[i])]
        #calculating the mindistance for each driver for this specific cluster
        minDistances = drivers.apply(lambda driver: nearestNeighbor(storesSubset, [driver[1], driver[2]], center), axis=1)
        #writing to output
        for j, driver in enumerate(drivers['DriverID']):
            writer.writerow([str(i), driver, minDistances[j]])

