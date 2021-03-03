import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage 
from sklearn.cluster import AgglomerativeClustering 
 


customer_data = pd.read_csv('shopping_data.csv')

print(customer_data.shape)

print(customer_data.head())

data = customer_data.iloc[:, 3:5].values

print(data.shape)

linked = linkage(data, 'single')
labelList = range(0, 199) 
plt.figure(figsize=(10, 7)) 
dendrogram(linked, orientation='top',  distance_sort='descending', show_leaf_counts=True) 
plt.show()

cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
cluster.fit_predict(data)

plt.scatter(data[:,0],data[:,1], c=cluster.labels_, cmap='rainbow') 
plt.show()
