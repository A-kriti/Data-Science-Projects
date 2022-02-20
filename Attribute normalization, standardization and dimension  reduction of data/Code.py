
# importing all the useful library fuction  
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale
import pandas as pd
import numpy as np
import math
import statistics
import matplotlib.pyplot as plt 

from sklearn.metrics import mean_squared_error

f = pd.read_csv("landslide_data3.csv")

df=pd.DataFrame(f) #forming data frame having all the attributes and values of it

# In[0]:

print("ques 1")
print("\n")

l=['temperature','humidity','pressure','rain','lightavgw/o0','lightmax','moisture']
def R_outliers(s):    #defining fun R_outliers(s)
    global X1               #assigning it as global variable for futher usages
    X=list(df[s])   #list of the s attributes from the data frame 
    
    
    

    
    t1=np.quantile(X,0.25)   #calculating the 1st quantile of s attribute
    t2=np.quantile(X,0.75)     #calculating the 3st quantile of s attribute
    
    k1=t1-(1.5*(t2-t1)) #calculating the value of lower whisker
    k2=t2+(1.5*(t2-t1))  #calculating the value of upper whisker
    
    c=0            #intialising the value of c
    l1=[]        #intialising the list l1 containg the index of outliers
    l2=[]            #intialising the list l2 containg all the values of s attributes which is not a outlier
    for i in range(len(X)):
        if X[i]<k1 or X[i]>k2:         #checking the values of ouliers by comparing with both upper and lower whisker
            l1.append(i)
            X[i]=math.nan      #assiging all the ouliers as nan
            c+=1
        else:
            l2.append(X[i])
    print("Total no of outliers in {0} is:".format(s),c)
    
    Y=np.median(l2)  #median of list with no ouliers
    for j in l1:
        X[j]=Y      #assiging all nan values as Y
        
    
    
    mean_before=np.mean(X)            #calculating mean and standard deviation of the intial data
    sd_before=statistics.stdev(X)
    
    min_X=min(X)
    max_X=max(X)
    
    print("_______________________ques a______________________________")
    print("The min value and max value of attribute '{0}' before normalisation:".format(s))
    print("min=",round(min_X,3)," ","max=",round(max_X,3))
   
    
    
    
   
    X1=X.copy()
    Z = minmax_scale(X, feature_range=(3,9))       #compressing the values to a fixed scale(3,9)
    min_Z=min(Z)
    max_Z=max(Z)
    print("The min value and max value of attribute '{0}' after normalisation:".format(s))
    print("min=",round(min_Z,3)," ","max=",round(max_Z,3))
    print("__________________________ques b___________________________")
    print("The mean and standard deviation of attribute '{0}' before normalisation:".format(s))
    print("mean=",round(mean_before,3)," ","Standard deviation=",round(sd_before,3))
    
    for i in range(len(X1)):
        X1[i]=(X1[i]-mean_before)/sd_before
    
    mean_after=np.mean(X1)    
    sd_after=statistics.stdev(X1)
    print("The mean and standard deviation of attribute '{0}' after normalisation:".format(s))
    print("mean=",round(mean_after,3)," ","Standard deviation=",round(sd_after,3))
    
    print("\n")
    return X1
    

l3=[]
l=['temperature','humidity','pressure','rain','lightavgw/o0','lightmax','moisture']
for i in l:
    s=i
    R_outliers(s)       #calculating all the required values for a particular attribute
  
    l3.append(X1)




# In[1]:
    

#_____________________________________________________________________________________
print('\n')
print("ques 2")

mean = [0, 0]
#cov = [[5, 10], [10, 13]]  # diagonal covariance

cov = [[6.84806467,7.63444163],[7.63444163,13.02074623]]         
x, y = np.random.multivariate_normal(mean, cov, 1000).T          #generating  2-dimensional synthetic data of 1000 samples 
D = np.random.multivariate_normal(mean, cov, 1000)

print('\n')
print("ques 2(a)_____________________________________")

plt.scatter(x, y,color='red',marker='x')
plt.axis('equal')
plt.title("Scattered plot of data samples")
plt.show()


print('\n')
print("ques 2(b)_______________________________________")

values, vectors = np.linalg.eig(cov)
print("Eigen vectors: ",vectors)
print("Eigen values: ",values)

#eig_vec1 = vectors[:,0]
#eig_vec2 = vectors[:,1]


plt.scatter(x, y,color='red',marker='x')
plt.axis('equal')
plt.title("Plot of 2D synthetic data and eigen direction")

plt.quiver(0,0, vectors[0],vectors[1],scale=9,color='k')

plt.show()


print('\n')
print("ques 2(c)_____________________________________")

v1=vectors.T[0]      #1st eigen vector
v2=vectors.T[1]          #2nd eigen vector

w=pd.DataFrame(D,columns=['w1','w2'])    #converting  2-dimensional synthetic data of 1000 samples into w dataframe

cordinates=np.asarray(list(zip(w['w1'],w['w2'])))               #ziping both the list in a single one
transformed_x=np.asarray([np.dot(v1, x1) for x1 in cordinates])     #doing the required calculation for the projection 
transformed_y=np.asarray([np.dot(v2, y1) for y1 in cordinates])
data=np.asarray(list(zip(transformed_x,transformed_y)))
transformed_data=pd.DataFrame(data=data,columns=['w1','w2'])
transformed_data=transformed_data.apply(lambda x: x.round(1))
plt.scatter(transformed_data['w1']*v1[0],transformed_data['w1']*v1[1],c='r',marker='x')
plt.axis('equal')
plt.show()



plt.scatter(transformed_data['w2']*v2[0],transformed_data['w2']*v2[1],c='blue',marker='x')
plt.show()

#plotting all the required graph 
plt.scatter(w["w1"],w['w2'],c='pink',marker='x')
plt.quiver(0,0, vectors[0],vectors[1],scale=9,color='blue')
plt.scatter(transformed_data['w1']*v1[0],transformed_data['w1']*v1[1],c='r',marker='x')
plt.title(" scatter plots of superimposed eigen vectors with 1st Eigen direction")
plt.show()



plt.scatter(w["w1"],w['w2'],c='pink',marker='x')
plt.quiver(0,0, vectors[0],vectors[1],scale=9,color='blue')
plt.scatter(transformed_data['w2']*v2[0],transformed_data['w2']*v2[1],c='r',marker='x')
plt.title(" scatter plots of superimposed eigen vectors with 2nd Eigen direction ")
plt.show()


print('\n')
print("ques 2(d)_____________________________________")


reconstruct1=np.dot(transformed_data,vectors)
total_var=w.var().sum()
ratio=transformed_data.var()/total_var

#calculating reconstruction error using formulas
b1=(w-transformed_data)**2 
b2=(np.sum(b1,axis=0)/1000)**0.5

print("Required reconstruction error: ","\n",b2)


# In[2]:


#_______________________________________________________________________________________________________
print("\n")
print("ques 3")
df1 = pd.DataFrame(np.array(l3).T,columns=['temperature','humidity','pressure','rain','lightavgw/o0','lightmax','moisture'])
#df1 is data frame formed by the outliers corrected standardized data
#print(df1.head())

pca = PCA(n_components=2)      #compressing it with 2 components
principalComponents = pca.fit_transform(df1)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])


print('\n')
print("ques 3(a)_______________________________________________")
#print(principalDf.head())
print("variance : ",pca.explained_variance_)
r1=principalDf.cov()

r2=list(principalDf['principal component 1'])
r3=list(principalDf['principal component 2'])

eigen_values, eigen_vectors = np.linalg.eig(r1)
print("Eigen values: ",eigen_values)
#print("Eigen vectors: ",eigen_vectors)
plt.scatter(r2,r3,c="orange",marker='x')
plt.title("Scatter plot of reduced dimensional data")
plt.show()

print('\n')
print("ques 3(b)_______________________________________________")

l=['temperature','humidity','pressure','rain','lightavgw/o0','lightmax','moisture']

t=df1.cov()
#print(len(t))

E_values, E_vectors = np.linalg.eig(t)

print("Required eigenvalue: ",E_values)

res = {E_values[i]:l[i] for i in range(len(E_values))} 
            
#print(res)

sorted_dict = sorted(res.items(),reverse=True)
#print(sorted_dict)
x, y = zip(*sorted_dict) # unpack a list of pairs into two tuples

#plotting the required eigen values of each attributes
plt.plot(x, y, color='green', marker='o')
plt.title("plot of eigen values of each attribute")
plt.show()



print('\n')
print("ques 3(c)_______________________________________________")

k=[]
for i in range(1,8):
        
    pca_1 = PCA(n_components=i)                  #n_components=2 using PCA
    principal_Components = pca_1.fit_transform(df1)
    reconstruct=pca_1.inverse_transform(principal_Components)
    principal_Df = pd.DataFrame(data = reconstruct)
    mse = mean_squared_error((df1),principal_Df)     #finding mse of (intial value,final value)
    rmse=math.sqrt(mse)                        #claculating rmse
    k.append(rmse)
 
#print(k)
#Ploting the RMSE v/s L for each attribute
m=[1,2,3,4,5,6,7]
plt.plot(m,k, color='purple', marker='o')
plt.xlabel("Value of L")
plt.ylabel("RMSE Values")
plt.title("Plot of RMSE v/s L for each attribute")
plt.show()

