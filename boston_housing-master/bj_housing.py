import  numpy as np
import pandas as pd
import visuals as vs

data = pd.read_csv('bj_housing.csv');
prices = data['Value']
features = data.drop('Value',axis=1)

from sklearn.cross_validation import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(features,prices,train_size=0.8,test_size=0.2,random_state=42)

import matplotlib.pyplot as plt
plt.scatter(features['Area'],prices,c='red')
plt.scatter(features['Room'],prices) 
from sklearn.metrics import r2_score
def performance_metric(y_true,y_predict):
    score = r2_score(y_true,y_predict) 
    return score
vs.ModelLearning(X_train,Y_train)
vs.ModelComplexity(X_train,Y_train)

from sklearn.model_selection import KFold,GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import make_scorer

def fit_Model(X,y):
    cross_validator = KFold(n_splits=10)
    reg = DecisionTreeRegressor()
    params = {"max_depth":range(1,11)}
    scoring_fnc = make_scorer(performance_metric)
    grid = GridSearchCV(reg,params,scoring_fnc,cv=cross_validator)
    grid.fit(X,y)
    #print grid.cv_results_
    return grid.best_estimator_

optimal_reg = fit_Model(X_train,Y_train)
print ("Parameter 'max_depth' is {} for the optimal model.".format(optimal_reg.get_params()['max_depth']))

client_data = [[128,3,1,1,2004,21],
               [84,2,2,0,2015,31],
               [120,3,2,1,2017,12]]
pridicted_price = optimal_reg.predict(client_data)
for i,prices in enumerate(pridicted_price):
    print ("Predicted selling price for Client {}'s home: RMB".format(i+1, prices))
    
