# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import Libraries: Load necessary libraries (Pandas, Scikit-learn).
2.Load Data: Read the employee data from a CSV file.
3.Explore Data: Display dataset information and check for missing values.
4.Preprocess Data: Encode categorical variables and define features (X) and target (y).
5.Split Data: Divide the data into training (80%) and testing (20%) sets.
6.Train Model: Initialize and fit a Decision Tree Classifier on the training data.
7.Make Predictions: Predict employee churn on the test set.
8.Evaluate Model: Calculate and display the model's accuracy.
```

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: EZHIL SREE J
RegisterNumber:  212223230056
*/
```
```
import pandas as pd
data=pd.read_csv("Employee.csv")
```
```
data.head()
```

## Output:
![image](https://github.com/user-attachments/assets/09d8d13e-62b3-4ca2-96fa-ffb487af5f5d)
```
data.info()
```

![image](https://github.com/user-attachments/assets/d209bb08-a0da-4fcd-9a80-e95f7655b91c)
```
data.isnull().sum()
```

![image](https://github.com/user-attachments/assets/cf329e90-4471-4bad-85b7-c00563538845)
```
data["left"].value_counts()

```

![image](https://github.com/user-attachments/assets/c1c17c0d-c48f-4f78-a6ee-bff15dcc5cb1)
```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
```

![image](https://github.com/user-attachments/assets/300d21d1-cf2d-4b19-9517-484d3f81dfa3)
```
x=data[["satisfaction_level", "last_evaluation", "number_project", "average_montly_hours", 
        "Work_accident", "promotion_last_5years", "salary"]]
x.head()
y=data["left"]
x.head()
```

![image](https://github.com/user-attachments/assets/e2c68fea-02ce-4a19-809f-dd464f8be327)
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
```
```
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,0,1,2]])
```
## Output:
![image](https://github.com/user-attachments/assets/650ac070-b930-422c-972d-5feba87fad61)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
