
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
#importing all the required libraries

f = pd.read_csv("seismic_bumps1.csv")
#f is a data frame formed using the given csv file
 
data=['seismic','seismoacoustic','shift','genergy','gpuls','gdenergy','gdpuls','ghazard','energy','maxenergy','class']  #list representing the required attributes
new_csv=f[data]
new_csv.to_csv("modified_data.csv",index=False) #converting modified file to a new csv file

# In[0]: 
print("ques 1")

f1 = pd.read_csv("modified_data.csv")   #reading the required csv file
df1 = pd.DataFrame(f1)         #data frame containing the modified csv file  
#print(df1.head())
Y=df1['class']         # Y is the data frame of class attribute
X_train, X_test,Y_train, Y_test= train_test_split(df1,Y,  test_size=0.3, random_state=42, shuffle=True)   #splitting the data into training and testing data

#print((X_train))


def knn(s):        #function giving all the required confusion and accuracy for all k
    knn = KNeighborsClassifier(n_neighbors=s)
#print(len(list(X_train)))
#print(len(list(X_test)))
    knn.fit(X_train, Y_train)     #fiting the data
    print("part 1(a)")
    print("\n")
    y_pred = knn.predict(X_test)         #prediction of the test data
    
    z=confusion_matrix(Y_test,y_pred)      #confusion matrix with the actual and predicted form
    print("Confusion matrix")
    print(z)
    
    #confusion_matrix(X_test,Y_test)
    print("part 1(b)")
    print("\n")
    print("Accuracy:",round(metrics.accuracy_score(Y_test, y_pred),3))  #printing the accuracy for every k
    
l=[1,3,5]       #list containing every k
for i in l:
    s=i 
    print("________________For k={0}________________".format(i))
    knn(s)     #calling the fuction for each k

# In[1]:    
#___________________________________________________________________________________________________________________________    
print("------------------------------------")

(X_train).to_csv("seismic-bumps-train.csv",index=False)   #conveting X_train into csv file
  

(X_test).to_csv("seismic-bumps-test.csv",index=False)    #conveting X_test into csv file





print("ques 2")

df_train=pd.read_csv("seismic-bumps-train.csv")
df_train.drop(['class'],axis=1,inplace=True)     #dropping the class attribute from train csv
df_test=pd.read_csv("seismic-bumps-test.csv")
df_test.drop(['class'],axis=1,inplace=True)         #dropping the class attribute from test csv
df_norm_train=(df_train-df_train.min())/(df_train.max()-df_train.min())    #normalising the train data
df_norm_test=(df_test-df_train.min())/(df_train.max()-df_train.min())      #normalising the test data accoring to the min max value of train data
 



def norm_knn(s1):         #fuction returing all the required confusion matrix and accuracy of normalised form for every k
    knn1 = KNeighborsClassifier(n_neighbors=s1)
    knn1.fit(df_norm_train,Y_train)       #fiting the data and making prediction
    pred1=knn1.predict(df_norm_test)
    print("part 2(a)")
    print(confusion_matrix(Y_test,pred1))
    print("part 2(b)")
    print("Accuracy:",round(metrics.accuracy_score(Y_test, pred1),6))    #required accuracing 

l=[1,3,5] #list containing every k
for i in l:
    s1=i 
    print("_________________For k={0}______________".format(i))
    norm_knn(s1) #calling the fuction for each k


# In[2]:
#____________________________________________________________________________________________
print("------------------------------------")
print("ques 3")


print('_______________Question3______________________')
X1_train=pd.read_csv('seismic_bumps_train.csv')
X1_test=pd.read_csv('seismic_bumps_test.csv')

C0=X1_train[X1_train['class']==0][X1_train.columns[0:-1]]     #finding the mean and covariance for class 0
Mean_C0=C0.mean().values;Cov_C0=C0.cov().values



C1=X1_train[X1_train['class']==1][X1_train.columns[0:-1]]
Mean_C1=C1.mean().values;Cov_C1=C1.cov().values        #finding the mean and covariance for class 1


P_C0=len(C0)/(len(C0)+len(C1))
P_C1=len(C1)/(len(C0)+len(C1))
d=len(X1_test.columns)-1

Predicted_class=[]            #appling all the operation for bayes classifier
for x in X1_test[X1_test.columns[0:-1]].values:
    
    p_x_C0=1/(((2*np.pi)**(d/2))*np.linalg.det(Cov_C0)**0.5)*np.e**(-0.5*np.dot(np.dot((x-Mean_C0).T,np.linalg.inv(Cov_C0)),(x-Mean_C0)))
    p_x_C1=1/(((2*np.pi)**(d/2))*np.linalg.det(Cov_C1)**0.5)*np.e**(-0.5*np.dot(np.dot((x-Mean_C1).T,np.linalg.inv(Cov_C1)),(x-Mean_C1)))
    P_x=p_x_C0*P_C0+p_x_C1*P_C1
    
    P_C0_x=p_x_C0*P_C0/P_x
    P_C1_x=p_x_C1*P_C1/P_x
    
    if P_C0_x>P_C1_x:Predicted_class.append(0)
    else:Predicted_class.append(1)
    
print('Confusion Matrix :\tAccuracy score :')
print(metrics.confusion_matrix(X1_test[X1_test.columns[-1]],Predicted_class),end='\n')
print("Accuracy:",round(metrics.accuracy_score(X1_test[X1_test.columns[-1]],Predicted_class),3),'\n')


# In[4]:
print("------------------------------------")
print("ques4")
# For KNN classifier
print("For KNN classifier ")
print("The best accuracy percent for k= 5 and it's value is :")
print(92.396, "%")
# For KNN classifier on normalized data
print("For KNN classifier  on normalized data")
print("The best accuracy percent  for k= 5 and it's value is :")
print(92.396, "%")
# For Bayes classifier using unimodal gaussian density
print("Fr Bayes Classifier ")
print("The best accuracy percentage is ")
print(88.917 , "%")
