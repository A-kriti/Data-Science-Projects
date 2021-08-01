
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import  metrics 
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import mixture

# In[0]:

print("Part A")
print("-------------------------------------------")
print("ques1") 
X_train = pd.read_csv('seismic-bumps-train.csv')
X_test = pd.read_csv('seismic-bumps-test.csv')

attr = list(X_train)[1:]
X_train0 = X_train[X_train['class']==0].loc[:,attr[:-1]]
X_train1 = X_train[X_train['class']==1].loc[:,attr[:-1]]

prior0 = len(X_train0)/len(X_train)
prior1 = len(X_train1)/len(X_train)

def GMM_model(q):    #fuction giving the required confusion matrix and accuracy for all values of q
    GMM = mixture.GaussianMixture(n_components=i,covariance_type='full',random_state=42)
    
    GMM0 = GMM.fit(X_train0)      #fitting the train data based on class 0
    log0 = GMM0.score_samples(X_test.loc[:,attr[:-1]])
    post0 = np.exp(log0)*prior0
    
    GMM1 = GMM.fit(X_train1)       #fitting the train data based on class 1
    log1 = GMM1.score_samples(X_test.loc[:,attr[:-1]])
    post1 = np.exp(log1)*prior1
    
    GMM_predict =[]
    for j in range(len(X_test)):   
        if post0[j] > post1[j]:      #comparing both post0 and post1 elements
            GMM_predict.append(0)
        else:
            GMM_predict.append(1)
    
    print("\nfor Q equal to:",i)
    print('Confusion Matrix :\n',confusion_matrix(X_test['class'],GMM_predict))        #printing the required matrix
    print('Accuracy =',accuracy_score(X_test['class'],GMM_predict))    #printing the required accuracy 

l=[2,4,8,16]          #list containing all the given values of q
for i in l:
       q=i
       GMM_model(q)        #calling the fuction for every q


# In[1]: 
       
       
print("-----------------------------------------")
print("ques2")
# For KNN classifier
print("For KNN classifier ")
print("The best accuracy percent for k= 5 and it's value is :")
print(92.396, "%")
# For KNN classifier on normalized data
print("For KNN classifier  on normalized data")
print("The best accuracy percent  for k= 5 and it's value is :")
print(92.396, "%")
# For Bayes classifier using unimodal gaussian density
print("For Bayes Classifier ")
print("The best accuracy percentage is ")
print(88.917 , "%")
# For Bayes classifier using GMM
print("For Bayes Classifier using GMM ")
print("The best accuracy percent is for q=8 ")
print(92.525 , "%")
print('\n')


# In[2]:
  
print("____________________________________________")
print("----------------------Part B-----------------------")
print("ques1")
print('\n')
f = pd.read_csv("atmosphere_data.csv")  #reading the given csv file
df = pd.DataFrame(f)           #data frame of the given csv file
x=df['pressure'].values.reshape(-1,1)              #reshaping pressure column
y=df['temperature'].values.reshape(-1,1)           #reshaping temperature column
#spliting the data into test and train data form
X_train, X_test,y_train,y_test = train_test_split(x,y , test_size=0.3, random_state=42, shuffle=True)     

  

#ques 1

regressor = LinearRegression()
regressor.fit(X_train, y_train) #fiting both the train data using linear regression
y_pred = regressor.predict(X_test)     #predicting the test data


print("-----------------------1 (a)--------------------------")
print("\n")
plt.scatter(X_test, y_test,  color='blue')        #plotting the required scattered plot
plt.plot(X_test, y_pred, color='red', linewidth=2)       #plotting the curve fitting plot 
plt.title('Linear Regression') 
plt.ylabel('Temperature') 
plt.xlabel('Pressure') 
plt.show()

y1_pred = regressor.predict(X_train)          #predicting the train data


print("----------------------------1 (b)------------------------")
print("\n")
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y1_pred))) #the required accuracy of the train data


print("------------------------1 (c)-----------------------------")
print("\n")
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred))) #the required accuracy of the test data



print("------------------------1 (d)-------------------")
print("\n")
plt.scatter(y_test, y_pred,  color='orange')        #scattered plot between the actual and predicted temperature
plt.xlabel("Actual temperature")
plt.ylabel("Predicted temperature on test data")
plt.title("scattered plot")
plt.show()


# In[3]:
#_______________________
#ques 2
print("-------------------------------------------")
print("ques2")


def nonlinear_regression(k):     #function returing the required values 
    global q1      #accuracy based on root mean square error of train data
    global q2       #accuracy based on the root mean square error of test data
    poly = PolynomialFeatures(degree = k) 
    X_poly = poly.fit_transform(X_train)      #fitting the train data into polynomial regression
  
    poly.fit(X_poly, y_train)     #fitting the above poly form and train data 
    lin2 = LinearRegression() 
    lin2.fit(X_poly,y_train)
    y2_pred = lin2.predict(poly.fit_transform(X_test)) 
    y3_pred = lin2.predict(poly.fit_transform(X_train)) 
    q1=np.sqrt(metrics.mean_squared_error(y_train, y3_pred))    #accuracy based on root mean square error of train data
    q2=np.sqrt(metrics.mean_squared_error(y_test, y2_pred))     #accuracy based on root mean square error of test data
    print("______________________a____________________")
    print('Root Mean Squared Error of train data:', np.sqrt(metrics.mean_squared_error(y_train, y3_pred)))  #the required accuracy of train data
    print("____________________b__________________________")
    print('Root Mean Squared Error of test data:', np.sqrt(metrics.mean_squared_error(y_test, y2_pred)))  #the required accuracy of test data
    print('\n')
    #the RMSE for p=5 is minimum resulting in the higher accuracy
    if k==5:    #plotting the fitting curve for p=5 
        print("____________________c__________________________")
        x = np.arange(30, 1100, 0.1).reshape(-1,1)
        polynomial_features = PolynomialFeatures(degree=k)
        x_poly = polynomial_features.fit_transform(X_train)#monomials for train data
        x1_poly = polynomial_features.fit_transform(x)
        lin2.fit(x_poly,y_train)
        y=lin2.predict(x1_poly)
        plt.plot(x,y,'r',label="predicted")
        plt.legend()
        plt.scatter(X_train, y_train, color = 'blue') 
        plt.title('Polynomial Regression for p=5') 
        plt.ylabel('Temperature') 
        plt.xlabel('Pressure') 
        plt.show()
        print('\n')
        #plotting the scattered plot for p=5
        print("____________________d__________________________")    
        plt.scatter(y_test, y2_pred,  color='green')
        plt.xlabel("Actual temperature")
        plt.ylabel("Predicted temperature on test data")
        plt.title("Scattered plot for p=5")
        plt.show()
        print('\n')
    
        
l=[2,3,4,5]  #list containing all the given values of p
l1=[]      #list conatining all the values of RMSE of train data
l2=[]        #list conatining all the values of RMSE of test data
for i in l:
    k=i
    nonlinear_regression(k)
    l1.append(q1)
    l2.append(q2)
    
#print(min(l1))
#print(min(l2))
#plotting the bar graph of train data 
print('\n')    
print("______________________plot a____________________")    
plt.bar(l,l1)
plt.title("Bar graph of Train data")
plt.xlabel("Degree of polynomial ")
plt.ylabel("RMSE  ")
plt.show()
#plotting the bar graph of test data 
print('\n')
print("______________________plot b____________________")
plt.bar(l,l2)
plt.title("Bar graph of Test data")
plt.xlabel("Degree of polynomial ")
plt.ylabel("RMSE  ")

plt.show()


