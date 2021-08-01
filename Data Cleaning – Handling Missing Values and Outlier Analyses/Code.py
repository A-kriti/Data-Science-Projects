
import pandas as pd
import numpy as np
#import statistics
import matplotlib.pyplot as plt

f = pd.read_csv("pima_indians_diabetes_miss.csv")

v=pd.DataFrame(f) #forming data frame having all the attributes and values of it


#q1
print("Ques 1")
print('\n')


x=v.isnull().sum()   #counting all the empty value and gives the total sum
print(" \nCount total NaN at each column in a DataFrame : \n\n",x ) 

x.plot.bar(rot=0, subplots=True)  #bargraph representing every attributes containg numper of empty space

plt.title("Bar graph repesenting frequency of empty space in each attribute")
plt.xlabel("Attributes")
plt.ylabel("count of NaN")
plt.show()
print('\n')

#q2
print("Ques 2(a)")
print('\n')

#(a)

z=list(v.isnull().sum(axis=1))  #list containing all the values NaN replaced by 0
c=0
h1=[]
for i in range(len(z)):
    if z[i]>=3:
        v.drop([i], inplace = True)
        c+=1
        h1.append(i+2)
 
print(h1)     # list containing row numbers of the deleted tuples with respect to the given csv file  
print("\n")
print("Total number of deleted tuple :",c) #total deleted tuple
print('\n')

#(b)

print("Ques 2(b)")
print("\n")

A=list(v[v["class"].isnull()].index) #finding index of all the missing values in the class attribute
v=v.drop(A) #droping the empty row
G=[]
for i in A:
    G.append(i+2)
print(G)  #list containing all the index of the empty row with respect to the given csv file





#printing total number of deleted tuple
print("Total number of deleted tuple having missing values in class attribute :",len(A))
print("\n")

#3
print("Ques 3")
print('\n')


x=v.isnull().sum()  #finding the number of empty rows left after the deletion of tuples in ques 2
print(x)   #the required count of all the missing values
print("\n")
print(" Total number of missing values in the file (after the deletion of tuples) :",sum(list(x)))
print('\n')

v1=v.copy()  #coping the dataframe v to use in ques 4
v2=v.copy()

#4

print("Ques 4(a)(i)")
print('\n')




#(a)(i)

r = pd.read_csv("pima_indians_diabetes_original.csv")
df1=pd.DataFrame(r)  #dataframe of the original data file


v1.fillna(v1.mean(),inplace=True)   # replacing values of all the empty attributes with there respective means


#print(df.head())
B1=v1.mean()     #finding mean of all the attributes of missing values
B2=df1.mean()    #finding mean of all the attributes of original values
C1=v1.median()    #finding median of all the attributes of missing values
C2=df1.median()    #finding median of all the attributes of original values
D1=v1.mode()       #finding mode of all the attributes of missing values
D2=df1.mode()       #finding mode of all the attributes of original values
E1=v1.std(axis = 0, skipna = True)    #finding standard deviation of all the attributes of missing values
E2=df1.std(axis = 0, skipna = True)    #finding standard deviation of all the attributes of original values



print("Mean of all the attributes of missing data using mean method is \n {0} \n Mean of all the attributes of original data is \n {1}".format(B1,B2))
print("Median of all the attributes of missing data using mean method is \n {0} \n Median of all the attributes of original data is \n {1}".format(C1,C2))
print("Mode of all the attributes of missing data using mean method is \n {0} \n Mode of all the attributes of original data is \n {1}".format(D1,D2))
print("Standard deviation of all the attributes of missing data using mean method is \n {0} \n Standard deviation of all the attributes of original data is \n {1}".format(E1,E2))
print('\n')




#(ii)
print("Ques 4(a)(ii)")
print('\n')



def fun(s,v,df1):   # fucntion giving the required rmse value for each attribute
    m1=list(v[v[s].isnull()].index)  #finding the index of the empty row of that particular s atribute
    #print(m1)
    na=0      #intiating count of na
    sq_sum=0   #intiating square sum 
    if m1==[]:
        return 0
    else:
        for i in m1:
            sq=(v1[s][i]-df1[s][i])**2   #step calculation
            sq_sum+=sq
            na+=1
        rmse=(sq_sum/na)**0.5  #appling formula of rmse
        return rmse
    
q1=[]  #intiating list q1 
attr=['pregs','plas','pres','skin','test','BMI','pedi','Age','class']   #list containg all the attributes
for i in attr:
    
    
   q1.append(fun(i,v,df1))
#q1 representing the list of the required rmse value of each attributes    
plt.bar(attr,q1)  #bar graph plotting
plt.title("RMSE using mean method")
plt.show()










print('\n')

#(b)(i)
print("Ques 4(b)(i)")
print('\n')



v2.fillna(v2.interpolate(),inplace=True)  # replacing values of all the empty attributes with there respective linearly interpolated values
B3=v2.mean()            #finding mean of all the attributes of missing values
B4=df1.mean()        #finding mean of all the attributes of original values
C3=v2.median()       #finding median of all the attributes of missing values
C4=df1.median()      #finding median of all the attributes of original values
D3=v2.mode()         #finding mode of all the attributes of missing values
D4=df1.mode()        #finding mode of all the attributes of original values
E3=v2.std(axis = 0, skipna = True)     #finding standard deviation of all the attributes of missing values
E4=df1.std(axis = 0, skipna = True)    #finding standard deviation of all the attributes of original  values



print("Mean of all the attributes of missing data using interpolation method is \n {0} \n Mean of all the attributes of original data is \n {1}".format(B3,B4))
print("Median of all the attributes of missing data using interpolation method is \n {0} \n Median of all the attributes of original data is \n {1}".format(C3,C4))
print("Mode of all the attributes of missing data using interpolation method is \n {0} \n Mode of all the attributes of original data is \n {1}".format(D3,D4))
print("Standard deviation of all the attributes of missing data using interpolation method is \n {0} \n Standard deviation of all the attributes of original data is \n {1}".format(E3,E4))
print('\n')

#(ii)

print("Ques 4(b)(ii)")
print('\n')



def fun1(s):             # fucntion giving the required rmse value for each attribute
    m2=list(v[v[s].isnull()].index)     #finding the index of the empty row of that particular s atribute
    
    na1=0          #intiating count of na
    sq1_sum=0        #intiating square sum 
    if m2==[]:
        return 0
    else:
        for i in m2:
            sq1=(v2[s][i]-df1[s][i])**2    #step calculation
            sq1_sum+=sq1
            na1+=1
        rmse1=(sq1_sum/na1)**0.5          #appling formula of rmse
        return rmse1
    
q2=[]      #intiating list q2
attr=['pregs','plas','pres','skin','test','BMI','pedi','Age','class']     #list containg all the attributes
for i in attr:
    
    
   q2.append(fun1(i))
#q2 representing the list of the required rmse value of each attributes    
plt.bar(attr,q2)  #bar graph plotting
plt.title("RMSE using interpolation method")
plt.show()



#5 (i)


#5
print("Ques 5(i)")
print('\n')

#(i)
X=list(v2["Age"])   #list of the Age attributes from the data frame used in ques 4(b) using interpolation method
Y=list(v2["BMI"])   #list of the BMI attributes from the data frame used in ques 4(b) using interpolation method

t1=np.quantile(X,0.25)   #calculating the 1st quantile of Age attribute
t2=np.quantile(X,0.75)     #calculating the 3st quantile of Age attribute
t3=np.quantile(Y,0.25)     #calculating the 1st quantile of BMI attribute
t4=np.quantile(Y,0.75)      #calculating the 3st quantile of BMI attribute

k1=t1-(1.5*(t2-t1))
k2=t2+(1.5*(t2-t1))
c2=0   #intiallizing c2
l1=[]
print("All outliers present in Age attribute")
for i in range(len(X)):
    if X[i]<=k1 or X[i]>=k2:    # appling the outlier condition 
        
        l1.append(X[i])
        c2+=1
print("Total no of outliers in Age attribute is :",c2)
print('\n')
print("list containing all the outliers in Age attribute")
print(l1)          #list containing all the outliers in Age attribute
print('\n')

print('\n')
k3=t3-(1.5*(t4-t3))
k4=t4+(1.5*(t4-t3))
c3=0     #intiallizing c2
print("All outliers present in BMI attribute")
l2=[]
for i in range(len(Y)):
    if Y[i]<=k3 or Y[i]>=k4:
        
        l2.append(Y[i])
        c3+=1
print("Total no of outliers BMI attribute is :",c3)
print('\n')
print("list containing all the outliers in BMI attribute")
print(l2)     #list containing all the outliers in BMI attribute
print('\n')

box_plot_data=[X,Y]       #list containing list of rain and moisture
plt.boxplot(box_plot_data,patch_artist=True,labels=['Age','BMI'])     #required box plot
plt.title("Boxplot of both Age and BMI")
plt.ylabel("Values of Age and BMI ")
plt.show() 



#(ii)
print("Ques 5(ii)")
print('\n')



X=list(v2["Age"])
w1=np.median(X)     #median of the attribute Age
Y=list(v2["BMI"])
w2=np.median(Y)     #median of the attribute BMI
for i in range(len(Y)):
    
    if X[i]<=k1 or X[i]>=k2:
        X[i]=w1      #replacing the value equal to outliers with median of Age
    if Y[i]<=k3 or Y[i]>=k4:
        Y[i]=w2         #replacing the value equal to outliers with median of BMI
        
box_plot_data=[X,Y]       #list containing list of rain and moisture
plt.boxplot(box_plot_data,patch_artist=True,labels=['Age','BMI'])     
plt.title("Boxplot of both Age and BMI after replacing outliers with median")
plt.ylabel("Values of Age and BMI ")
plt.show() 

