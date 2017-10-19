import numpy as np
import pandas as pd
import visuals as vs # Supplementary code

from sys import version_info
if version_info.major != 2 and version_info.minor != 7:
    raise Exception()
    
data = pd.read_csv('housing.csv')
prices = data['MEDV']
features = data.drop('MEDV', axis = 1)
    

print "Boston housing dataset has {} data points with {} variables each.".format(*data.shape)

#TODO 1

minimum_price = np.min(prices)

maximum_price = np.max(prices)

mean_price = np.mean(prices)

median_price = np.median(prices)

std_price = np.std(prices)

print "Statistics for Boston housing dataset:\n"
print "Minimum price: ${:,.2f}".format(minimum_price)
print "Maximum price: ${:,.2f}".format(maximum_price)
print "Mean price: ${:,.2f}".format(mean_price)
print "Median price ${:,.2f}".format(median_price)
print "Standard deviation of prices: ${:,.2f}".format(std_price)

#Splite Datas

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features,prices,train_size=0.8)

def performance_metric(y_true, y_predict):
    
    score = r2_score(y_true,y_predict)

    return score
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer
from sklearn.tree import DecisionTreeRegressor
from sklearn.grid_search import GridSearchCV

def fit_model(X, y):
        
    cross_validator = KFold(n_splits=10, random_state=None, shuffle=False)
    
    regressor = DecisionTreeRegressor()

    params = {'max_depth':[1,2,3,4,5,6,7,8,9,10]}

    scoring_fnc = make_scorer(performance_metric)

    grid = GridSearchCV(estimator = regressor,param_grid = params,scoring_fnc=scoring_fnc, cv=cross_validator)

    grid = grid.fit(X, y)

    return grid.best_estimator_
# Fit the training data to the model using grid search
reg = fit_model(X_train, y_train)
# Produce the value for 'max_depth'
print "Parameter 'max_depth' is {} for the optimal model.".format(reg.get_params()['max_depth'])
#display(reg.get_params()['max_depth'])
