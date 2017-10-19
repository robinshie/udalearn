
import numpy as np
import pandas as pd
import visuals as vs
from IPython.display import display 


try:
    data = pd.read_csv("customers.csv")
    data.drop(['Region', 'Channel'], axis = 1, inplace = True)
    print "Wholesale customers dataset has {} samples with {} features each.".format(*data.shape)
except:
    print "Dataset could not be loaded. Is the dataset missing?"
display(data.describe())

indices = [0,data.shape[0]/2,data.shape[0]/2-1]


samples = pd.DataFrame(data.loc[indices], columns = data.keys()).reset_index(drop = True)
print "Chosen samples of wholesale customers dataset:"
display(samples)

dropList = np.array(['Grocery','Milk'])
from sklearn.tree import DecisionTreeRegressor
for i in range(dropList.shape[0]):
    
    new_data = data.drop(dropList[i], axis = 1)

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test =train_test_split(new_data, data[dropList[i]], test_size = 0.25, random_state = 0)

   
    regressor = DecisionTreeRegressor(random_state = 0)
    regressor=regressor.fit(X_train,y_train)

    score = regressor.score(X_test,y_test)
    print score