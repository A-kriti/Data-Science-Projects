
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from statsmodels.tsa.ar_model import AutoReg
import numpy as np
from sklearn.metrics import mean_squared_error


f = pd.read_csv("datasetA6_HP.csv")  #reading the given file
df=pd.DataFrame(f)       #converting f into data frame
l=list(df["HP"])   # list consisting of the elements of the 1st row of the given csv file


print("---------------ques1---------------")
print("\n")
print("____________ques1(a)_______________")
print("\n")
#1(a)
plt.plot(l,color="red")   #plotting the required line plot
plt.xlabel(" Index of the day")
plt.ylabel("Power consumed in megaunits (MU)")
plt.title("Line plot between Day's vs Power Consumption")
plt.show()


#1(b)
print("____________ques1(b)_______________")
print("\n")
def auto_correlation(k,lag):   #fuction giving the required auto correlation
    auto_corr=sm.tsa.acf(k,nlags=lag)   #correlation calculation
    return auto_corr        #returning the required correlation

prsn_corr=auto_correlation(f["HP"],1)      #finding the correlation for lag = 1
print("The required correlation is :",prsn_corr[1]) #printing the required correlation

#1(c)
print("____________ques1(c)_______________")
print("\n")
xt=[l[i] for i in range(1,500)]  #list containing the present values
xt1=[l[i] for i in range(0,499)]   #list containing the past values
plt.scatter(xt, xt1,color='purple')          #scattered plot of both y(t) and y(t-1)
plt.xlabel("Given time sequence")
plt.ylabel("Generated one day time lag sequence")
plt.show()

#1(d)
print("____________ques1(d)_______________")
print("\n")
corr=[] #list containing correlation of all lags
k1=[int(i) for i in range(1,8)]  #list containg all the required values of lag
for i in range(1,8):
    prsn_corr=auto_correlation(f["HP"],i)   #calling the correlation fuction to calculate correlation for every lag
    corr.append(prsn_corr[i])   #appending the required correlation correspond to every lag
#print(corr)
plt.plot(k1,corr, color="blue",linestyle="dashed",linewidth=2,marker="o")    #line plot of lag v/s correlation 
plt.xlabel("Time lags")
plt.ylabel("Pearson correlation coefficient")
plt.show() 
    
#1(e)  
print("____________ques1(e)_______________")
print("\n")  
sm.graphics.tsa.plot_acf(f["HP"], lags=7)    #by using direct method
plt.xlabel("Time lags")
plt.ylabel("Pearson correlation coefficient")
plt.show() 
    
#2
print("---------------ques2---------------")
print("\n")
test = l[len(l)-250:]    #test data consist last 250 values
Actual = test[:len(test)-1]   #consist of 1st 250 values
predicted = test[1:]        
rmse=(mean_squared_error(Actual,predicted))**0.5   #rmse calculation between acutal and predicted data
print("RMSE of persistance model is:",rmse)

print("\n")   
    
#3
print("---------------ques3---------------")
print("\n")
#3(a)   
print("____________ques3(a)_______________")
print("\n")

    

train, test = l[0:len(l)-250], l[len(l)-250:]    #splitting the data into test and train
# train autoregression
model = AutoReg(train, lags=5)   #creating model for lag=5
model_fit = model.fit()   #fitting the model
 
# make predictions
predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
#for i in range(len(predictions)):
    #print('predicted=%f, expected=%f' % (predictions[i], test[i]))
rmse = np.sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)    
        
plt.scatter(test,predictions)      #scattered plot between tested and predicted values
plt.xlabel("Original test stats")
plt.ylabel("predicted test stats")  
plt.show()  
    
    
#3(b) 
print("____________ques3(b)_______________")
print("\n")
def Auto_reg(i):      #fuction giving the required auto regression values for each lag
    model = AutoReg(train, lags=i)     #auto regression for each lag value
    model_fit = model.fit()   #fitting the model

    # make predictions
    predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
    rmse = np.sqrt(mean_squared_error(test, predictions))  #calculating the required rmse between tested and predicted values
    print('Test RMSE:  for lag = {0} :{1}'.format(i,rmse))    
        
    
p=[1,5,10,15,25]    #list containing all the given lag values
 
for j in p:
    i=j
    Auto_reg(i)   #calculating the auto regression for every lag value
    

#3(c)
print("\n")
print("____________ques3(c)_______________")
print("\n")
Y=f['HP'].values 
thres_val=2/(len(train)**0.5)         #calculating the heuristics value
auto_cor=np.corrcoef(Y[1:],Y[:-1])[1,0]   #caculating the auto correlation
corr=[]   #list containing all the correlation value
i=2        #putting i as 2
while abs(auto_cor)>thres_val:  #comparing with the threshold value
    corr.append(auto_cor)
    auto_cor=np.corrcoef(Y[i:],Y[:-i])[1,0]
    i=i+1    #incrementing i by 1
    
optimal_p=len(corr)   #len of the list corr
model = AutoReg(train, lags=optimal_p)   #model with optimal lag value
model_fit = model.fit()   #fitting the model

    # make predictions
predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
rmse_opt = np.sqrt(mean_squared_error(predictions,test))  #calculating the required rmse between tested and predicted values
#print("Test RMSE:  ",rmse_opt) 
print("So, optimal value of lag using heuristicsis is ",len(corr),"with rmse",rmse_opt)
print("The heuristic value for optimal number of lags is ",thres_val)   
        
print("____________ques3(d)_______________")
print("\n")

print("The optimal number of lags without using heuristics for calculating optimal lag is: 10 with rmse =4.526283621756578")
print("\n")
print("The optimal number of lags using heuristics for calculating optimal lag is:5 with rmse = 4.537007584381682")
