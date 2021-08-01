

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from scipy import spatial as spatial




f = pd.read_csv('mnist-tsne-train.csv')
g = pd.read_csv("mnist-tsne-test.csv")
df_train=pd.DataFrame(f)
df_test=pd.DataFrame(g)


X = np.array([list(f['dimention 1']),list(f['dimension 2'])])   #array consisting only dimenstion 1 and 2 of train data
#print(X.shape)
#print(X.shape)
Y = np.array([list(g['dimention 1']),list(g['dimention 2'])])   #array consisting only dimenstion 1 and 2 of test data
#print(X.shape)
#print(Y.shape)

# In[0]:
def purity_score(y_true, y_pred):
         # compute contingency matrix (also called confusion matrix)
         contingency_matrix=metrics.cluster.contingency_matrix(y_true, y_pred)
         #print(contingency_matrix)
         # Find optimal one-to-one mapping between cluster labels and true labels
         row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
         # Return cluster accuracy
         print(contingency_matrix[row_ind,col_ind].sum()/np.sum(contingency_matrix))
    
("--------------------------------------------------Question 1-------------------------------------------------------")
def k_means(K):
    global distortion

    kmeans = KMeans(n_clusters=K,random_state=(42))
    kmeans.fit(X.T)
    kc=kmeans.cluster_centers_
    kmeans_prediction_train = kmeans.predict(X.T)
    #print(kmeans_prediction)
    distortion = kmeans.inertia_
    
    print('1(i)')
    
    plt.scatter(X.T[:,0],X.T[:,1], c=kmeans.labels_, cmap='rainbow',marker = '*')
    plt.scatter(kc[:,0],kc[:,1],s = 80,c = 'black', marker = 'o') 
    plt.xlabel("dimension 1")
    plt.ylabel("dimension 2")
    title = "k means clustetrring of train data with k value " + str(K)
    plt.title(title)
    
    plt.show()
    
    
    print('1(ii)')
    
    
    
    print("Purity Score of the train data using Kmeans for K = {0} is given as:".format(K))
    purity_score(df_train["labels"], kmeans_prediction_train)
    
    
    
    print('1(iii)')
    
    
    
    kmeans = KMeans(n_clusters=K,random_state=(42))
    kmeans.fit(X.T)    #l[0:len(l)-250]
    kc_test=kmeans.cluster_centers_
    kmeans_prediction_test = kmeans.predict(Y.T)
    predict = kmeans_prediction_test
    #print(kmeans_prediction)
    
    
    
    plt.scatter(Y.T[:,0],Y.T[:,1], c=predict, cmap='rainbow',marker = '*')
    plt.scatter(kc_test[:,0],kc_test[:,1],s = 80,c = 'black', marker = 'o') 
    plt.xlabel("dimension 1")
    plt.ylabel("dimension 2")
    title = "k means clustetrring of test data with k value " + str(K)
    plt.title(title)
    #plt.legend()
    plt.show()
    
    print('1(iv)')
    
    print("Purity Score of the test data using Kmeans for K = {0} is given as:".format(K))
    purity_score(df_test["labels"], kmeans_prediction_test)
    
K = 10    
k_means(K)

# In[1]:
("--------------------------------------------------Question 2-------------------------------------------------------")
#ques 2
print("\n")
print("ques 2")


def Gmm_clustering(K):
    global log_likelihood
    gmm = GaussianMixture(n_components = K,random_state=(42))
    gmm.fit(X.T)
    GMM_prediction_train = gmm.predict(X.T)
    #print(GMM_prediction)
    centers_gmm = gmm.means_
    log_likelihood = gmm.lower_bound_
    
       
    print('2(i)')
    plt.scatter(X.T[:,0],X.T[:,1], c=gmm.fit_predict(X.T), cmap='rainbow',marker = '*')
     
    plt.scatter(centers_gmm[:,0],centers_gmm[:,1],c='black',s=80,alpha=0.7)
    plt.xlabel("dimension 1")
    plt.ylabel("dimension 2")
    title = "GMM clustetrring of train data with k value " + str(K)
    plt.title(title)
    plt.show()
        
    print('2(ii)')
    
    
    
    print("Purity Score of the train data using GMM for K = {0} is given as:".format(K))
    purity_score(df_train["labels"], GMM_prediction_train)
    
    print('2(iii)')
    gmm_test = GaussianMixture(n_components = K,random_state=(42))
    gmm_test.fit(X.T)
    GMM_prediction_test = gmm_test.predict(Y.T)
    #print(GMM_prediction)
    centers_gmm_test = gmm_test.means_
    
    
       
   
    plt.scatter(Y.T[:,0],Y.T[:,1], c=  GMM_prediction_test, cmap='rainbow',marker = '*',label = 'center')
    #[matplotlib.cm.spectral(float(i) /10) for i in cluster.labels_]
    plt.scatter(centers_gmm_test[:,0],centers_gmm_test[:,1],c='black',s=80,alpha=0.7)
    plt.legend()
    plt.xlabel("dimension 1")
    plt.ylabel("dimension 2")
    title = "GMM clustetrring of test data with k value " + str(K)
    plt.title(title)
    plt.show()
     
    print('2(iv)')
    print("Purity Score of the test data using GMM for K = {0} is given as:".format(K))
    purity_score(df_test["labels"], GMM_prediction_test)

K = 10
Gmm_clustering(K)


# In[2]:
print("\n")
print("ques 3")
("--------------------------------------------------Question 3-------------------------------------------------------")

print('3(i)')

def dbscan(epsilon,min_sample):
    dbscan_model=DBSCAN(eps=epsilon, min_samples=min_sample).fit(X.T)
    DBSCAN_predictions = dbscan_model.labels_
    plt.scatter(X.T[:,0], X.T[:,1], c=dbscan_model.labels_, cmap='rainbow',marker = '*',label = 'center')
    plt.xlabel("dimension 1")
    plt.ylabel("dimension 2")
    title = "DBSCAN of train data with epsilon " + str(epsilon) +" "+ "and min_samples " + str(min_sample)
    plt.title(title)
    plt.show()
    
    print('3(ii)')
    print("\n")
    print("Purity Score of the train data using DBSCAN for epsilon = {0} and min_sample = {1} is given as:".format(epsilon,min_sample))
    purity_score(df_train["labels"], DBSCAN_predictions)
    
    
    print('3(iii)')
    def dbscan_predict(dbscan_model, X_new, metric=spatial.distance.euclidean):
        # Result is noise by default
        y_new = np.ones(shape=len(X_new), dtype=int)*-1 
        # Iterate all input samples for a label
        for j, x_new in enumerate(X_new):
            # Find a core sample closer than EPS
            for i, x_core in enumerate(dbscan_model.components_):
                if metric(x_new, x_core) < dbscan_model.eps:
                    # Assign label of x_core to x_new
                    y_new[j] = dbscan_model.labels_[dbscan_model.core_sample_indices_[i]]
                 
                    break
        return (y_new)
    #dbtest = dbscan_predict(dbscan_model, DBSCAN_predictions, metric = spatial.distance.euclidean)
      
    
    
    
    
    dbscan_model_test=DBSCAN(eps=epsilon, min_samples=min_sample).fit(X.T)
   
    DBSCAN_predictions_test = dbscan_predict(dbscan_model_test,Y.T)
    plt.scatter(Y.T[:,0], Y.T[:,1], c=DBSCAN_predictions_test, cmap='rainbow',marker = '*',label = 'center')
    plt.xlabel("dimension 1")
    plt.ylabel("dimension 2")
    title = "DBSCAN of test data with epsilon " + str(epsilon) +" "+ "and min_samples " + str(min_sample)
    plt.title(title)
    plt.show()
    
    
    print('3(iv)')
    print("\n")
    print("Purity Score of the test data using DBSCAN for epsilon = {0} and min_sample = {1} is given as:".format(epsilon,min_sample))
    purity_score(df_test["labels"], DBSCAN_predictions_test)
    
epsilon=5
min_sample=10
dbscan(epsilon,min_sample)

# In[3]:
print("--------------------------------------------------Bonus Question-------------------------------------------------------")
print("\n")
print("_______________________________________________________Bonus Ques-1______________________________________________________")
print("\n")


print("Elbow method using K-means Clustering")
print("\n")
print("for other values of k")



l=[2,5,8,12,18,20]
l1=[]
for i in l:
    K=i
    k_means(K)
    l1.append(distortion)

#print("len l1:",l1)

plt.plot(l,l1,c='blue',linestyle="dashed",marker="o")
plt.xlabel("Values of K")
plt.ylabel("Distortion")
plt.title("Elbow Method for K-means Clustering")
plt.show()


print("Elbow method using GMM Clustering")
print("\n")
print("for other values of k")

l=[2,5,8,12,18,20]
l2=[]
for i in l:
    K=i
    Gmm_clustering(K)
    l2.append(log_likelihood)

#print("len l1:",l2)

plt.plot(l,l2,c='green',linestyle="dashed",marker="o")
plt.xlabel("Values of K")
plt.ylabel("log likelihood")
plt.title("Elbow Method for Clustering using GMM")
plt.show()


# In[4]:

print("\n")
print("_______________________________________________________Bonus Ques-2______________________________________________________")
print("\n")



h1=[1,5,10]  #epsilon value
h2=[1,10,30,50]   #min sample value

for i in h1:
    print("for value of min_sample = 10 and putting the value of epsilon = {0}".format(i))
    epsilon=i
    min_sample=10
    dbscan(epsilon,min_sample)
    
    
for i in h2:
    print("for value of epsilon = 5 and putting the value of min_sample = {0}".format(i))
    epsilon=5
    min_sample=i
    dbscan(epsilon,min_sample)
    
