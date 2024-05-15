# Implementation of SVM For Spam Mail Detection

## Aim:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the necessary packages.

2.Read the given csv file and display the few contents of the data.

3.Assign the features for x and y respectively.

4.Split the x and y sets into train and test sets.

5.Convert the Alphabetical data to numeric using CountVectorizer.

6.Predict the number of spam in the data using SVC (C-Support Vector Classification) method of SVM (Support vector machine) in sklearn library.

7.Find the accuracy of the model.
## Program:
```
Program to implement the SVM For Spam Mail Detection
Developed by:  Sathish R
RegisterNumber:  212222230138
```
```PYTHON
import chardet
file="/content/spam.csv"
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result
import pandas as pd
data=pd.read_csv("/content/spam.csv",encoding="Windows-1252")
data.head()
data.info()
data.isnull().sum()
x=data["v1"].values
y=data["v2"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
#### Result output
![Screenshot 2023-06-03 175457](https://github.com/Yamunaasri/Implementation-of-SVM-For-Spam-Mail-Detection/assets/115707860/7063a16e-46a2-4dbd-873c-c34b28b32532)

#### data.head()
![Screenshot 2023-06-03 175546](https://github.com/Yamunaasri/Implementation-of-SVM-For-Spam-Mail-Detection/assets/115707860/e6ffe7d7-9626-4670-a59c-aade7210a7e6)

#### data.info()
![Screenshot 2023-06-03 175721](https://github.com/Yamunaasri/Implementation-of-SVM-For-Spam-Mail-Detection/assets/115707860/4311d78e-102f-4d5d-817f-7b2615c69165)

#### data.isnull().sum()
![Screenshot 2023-06-03 175739](https://github.com/Yamunaasri/Implementation-of-SVM-For-Spam-Mail-Detection/assets/115707860/b3b9a6ec-aa9f-42c6-934f-6bde27e20bce)

#### Y_prediction value
![Screenshot 2023-06-03 175808](https://github.com/Yamunaasri/Implementation-of-SVM-For-Spam-Mail-Detection/assets/115707860/3c2c208a-bf0e-4e58-91ba-039fa57aa02c)

#### Accuracy value
![Screenshot 2023-06-03 175813](https://github.com/Yamunaasri/Implementation-of-SVM-For-Spam-Mail-Detection/assets/115707860/b00cb652-c51a-496c-a40d-45e3ccbef92d)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
