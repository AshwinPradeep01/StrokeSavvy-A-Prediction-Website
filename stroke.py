import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split

import warnings
df= pd.read_csv("train2.csv")

X = df.copy()
X = X.drop("stroke", axis=1)
X = X.iloc[:,:].values 

Y = df["stroke"]
    
    #test and train split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25,random_state=120)

    #objectification
clf = svm.SVC(probability=True)

    #training
clf.fit(X_train, Y_train)

 
 
        
