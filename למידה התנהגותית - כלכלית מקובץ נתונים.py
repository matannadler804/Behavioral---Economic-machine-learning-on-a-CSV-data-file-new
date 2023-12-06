# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

file_path = r"Family Income and Expenditure.csv"
data = np.genfromtxt(file_path,delimiter=',',skip_header=1)

numrows,numcul = data.shape 
print("number of rows:" , numrows)
print("number of columns",numcul)

mini_data=data[:,0:2]

norm_min_data=(mini_data-np.min(mini_data,axis=0))/(np.max(mini_data,axis=0)-np.min(mini_data,axis=0))

k_values = np.arange(1,16)
sse_value = []
for i in k_values:
    kmeans=KMeans(n_clusters=i,random_state=0).fit(norm_min_data)
    sse_value.append(kmeans.score(norm_min_data)*-1)
plt.plot(k_values,sse_value,'bo-')
plt.xlabel("number of clusters")
plt.ylabel("SSE")
plt.title("elbow graph")
plt.show()

optimal_k=None
for i in range(1,len(sse_value)-1):
    if sse_value[i]-sse_value[i-1]<=0.1*(sse_value[i-1]-sse_value[i-2]):
        optimal_k=i+1
        break
    
if optimal_k is None:
    optimal_k=np.argmin(np.diff(sse_value))+1

print("optimal k=",optimal_k)

k=3
kmeans=KMeans(n_clusters=k,random_state=0).fit(norm_min_data)
cluster_centers = kmeans.cluster_centers_
sse=kmeans.score(norm_min_data)*-1
print("cluster centers:")
print(cluster_centers) 
print("sse:")
print(sse)

plt.scatter(data[:,1],data[:,0],c=kmeans.labels_,cmap='viridis')
plt.xlabel('Total Food Expenditure')
plt.ylabel('Total Household Income')
plt.title('scatter plot of Total Food Expenditure vs Total Household Income')
plt.show()
print("גרף 1:")
print("קבוצה ראשונה מאופיינת בהוצאה נמוכה על אוכל ")
print("קבוצה שנייה מאופיינת בהוצאה גבוהה יותר מהקבוצה הראשונה  על אוכל ")
print("קבוצה שלישית  מאופיינת בהוצאה הגבוהה ביותר על אוכל ")

plt.scatter(data[:,9],data[:,0],c=kmeans.labels_,cmap='viridis')
plt.xlabel('Total Restaurant and hotels Expenditure')
plt.ylabel('Total Household Income')
plt.title('scatter plot of Total Restaurant and hotels Expenditure vs Total Household Income')
plt.show()

print("גרף 2:")
print("ניתן להסיק מהגרף שמי שמרוויח הרבה גם יוציא הרבה על מסעדות ומלונות")

plt.scatter(data[:,10],data[:,0],c=kmeans.labels_,cmap='viridis')
plt.xlabel('Total Alcoholic Beverages Expenditure')
plt.ylabel('Total Household Income')
plt.title('scatter plot of Total Alcoholic Beverages Expenditure vs Total Household Income')
plt.show()

print("גרף 3:")
print("ניתן להסיק מהגרף כי משפחות מבוססות קונות פחות אלכוהול ממשפחות שמרוויחות פחות")


plt.scatter(data[:,15],data[:,0],c=kmeans.labels_,cmap='viridis')
plt.xlabel('Total Medical Care Expenditure')
plt.ylabel('Total Household Income')
plt.title('scatter plot of Total Medical Care Expenditure vs Total Household Income')
plt.show()

print("גרף 4:")
print("ניתן להסיק מהגרף כי משפחות פחות מבוססות נוטות לרוב לחלות יותר ובכך לצרוך יותר תרופות ")




