from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error,r2_score
import pandas as pd
import numpy as np
import matplotlib as plt

df_=pd.read_csv("C:\\Users\\berat\\pythonEğitimleri\\python\\Turkcell Makine Öğrenmesi\\sınıflandırma\\diabetes.csv")
y=df_["Outcome"]
df=df_.drop(["Outcome"],axis=1)
x_train,x_test,y_train,y_test=train_test_split(df,y,test_size=0.3,random_state=200)

RFmodel=RandomForestClassifier()
RF_params={
    "n_estimators":[100,200,300,400,1000],
    "max_features":[1,5,7,8,10],
    "min_samples_split":[2,3,4,5,6]
}

RF_cv=GridSearchCV(RFmodel,RF_params,cv=5,n_jobs=-1,verbose=2)
RF_cv.fit(x_train,y_train)
n_estimators=RF_cv.best_params_["n_estimators"]
max_features=RF_cv.best_params_["max_features"]
min_samples_split=RF_cv.best_params_["min_samples_split"]

RF_tuned=RandomForestClassifier(n_estimators=n_estimators,
                                max_features=max_features,
                                min_samples_split=min_samples_split)
RF_tuned.fit(x_train,y_train)
predict=RF_tuned.predict(x_test)
acscore=accuracy_score(y_test,predict)
print(acscore)

#değişkenlerin önem sırasını verir
feature_imp=pd.Series(RF_tuned.feature_importances_,
                      index=x_train.columns.sort_values(ascending=False))














