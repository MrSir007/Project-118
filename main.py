import csv
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans as km
import matplotlib.pyplot as plt
import seaborn as sb

getData = pd.read_csv("stars.csv")

scatter = px.scatter(getData, x="Size", y="Light")
'''scatter.show()'''

x = getData.iloc[:,[0,1]].values
wcss = []
for i in range (1,11) :
  kmean = km(n_clusters=i,init="k-means++",random_state=42)
  kmean.fit(x)
  wcss.append(kmean.inertia_)
print(x)
plt.figure(figsize=(10,5))
sb.lineplot(range(1,11),wcss,marker='o',color="blue")
plt.title("The Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
'''plt.show()'''

kmean = km(n_clusters=3,init="k-means++",random_state=42)
y_kmean = kmean.fit_predict(x)

plt.figure(figsize=(15,7))
sb.scatterplot(x[y_kmean==0,0],x[y_kmean==0,1],color="blue",label="Cluster 1")
sb.scatterplot(x[y_kmean==1,0],x[y_kmean==1,1],color="red",label="Cluster 2")
sb.scatterplot(x[y_kmean==2,0],x[y_kmean==2,1],color="green",label="Cluster 3")
sb.scatterplot(kmean.cluster_centers_[:,0],kmean.cluster_centers_[:,1],color="black",label="centroid",s=100,marker=",")
plt.grid(False)
plt.title("Clusters of Stars")
plt.xlabel("Size")
plt.ylabel("Light")
plt.legend()
plt.show()