import pandas as pd 
import csv
import plotly.express as px
from sklearn.cluster import KMeans
import matplotlib.pyplot as mp
import seaborn as sb

df = pd.read_csv('data.csv')
light = df['Light'].tolist()
size = df['Size'].tolist()

fig=px.scatter(x=light,y=size)
X =  df.iloc[: , [0,1]].values
wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

mp.figure(figsize=(10,5))
sb.lineplot(range(1,11),wcss,marker="o",color="blue")

mp.title('Elbow Method')
mp.xlabel('Number of clusters')
mp.ylabel('WCSS')
#mp.show()

kmeans = KMeans(n_clusters=3,init='k-means++',random_state=42)
y_kmeans = kmeans.fit_predict(X)

mp.figure(figsize=(15,7))

sb.scatterplot(X[y_kmeans == 0,0] , X[y_kmeans == 0,1] , color='yellow' , label = 'Cluster 1')
sb.scatterplot(X[y_kmeans == 1,0] , X[y_kmeans == 1,1] , color='green' , label = 'Cluster 2')
sb.scatterplot(X[y_kmeans == 2,0] , X[y_kmeans == 2,1] , color='blue' , label = 'Cluster 3')
sb.scatterplot(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], color = 'red' , label='Center Anstreoinds',s=100,marker=',')
mp.grid(False)

mp.title('Clusters of Bulb')
mp.xlabel('Light')
mp.ylabel('Size')
mp.legend()
mp.show()