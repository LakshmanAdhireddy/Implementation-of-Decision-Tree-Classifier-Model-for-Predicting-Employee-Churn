# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries.

2.Upload and read the dataset.

3.Check for any null values using the isnull() function.

4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

5.Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Lakshman
RegisterNumber:  212222240001
*/

import pandas as pd
data=pd.read_csv("/content/Employee.csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])

```

## Output:

### Data head():
![image](https://github.com/LakshmanAdhireddy/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118707265/0702cfb6-91ab-4c76-9635-8bc9e056896d)

### Data set info():
![image](https://github.com/LakshmanAdhireddy/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118707265/c96b7bcc-8725-4b80-ad7b-6e83fb15a8fd)

### Null dataset:
![image](https://github.com/LakshmanAdhireddy/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118707265/1a89869e-a069-46b4-9a65-afe1a838df42)

### Values count():
![image](https://github.com/LakshmanAdhireddy/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118707265/3c2813f5-304e-49be-ab37-82ad7a3c0da2)

### Data head() for salary:
![image](https://github.com/LakshmanAdhireddy/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118707265/23fd4146-0efd-45cc-af5d-18f733aa61ba)

### x.head():
![image](https://github.com/LakshmanAdhireddy/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118707265/dd06b265-213c-4c4b-a2a7-3e903959f6dc)

### Accuracy value:
![image](https://github.com/LakshmanAdhireddy/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118707265/fcee4aac-2c2d-4a72-bf5f-c3b111ea82a6)

### Data prediction:
![image](https://github.com/LakshmanAdhireddy/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118707265/52086ab3-d88e-49c7-87e9-39298a9271e8)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
