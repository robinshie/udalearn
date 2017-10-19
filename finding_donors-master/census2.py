import numpy as np
import pandas as pd
from time import time
from IPython.display import display 


import visuals as vs

data = pd.read_csv("census.csv")

display(data.head(n=1))


n_records = data.shape[0]

n_greater_50k = data[data.income.str.contains('>50K')].shape[0]


n_at_most_50k = data[data.income.str.contains('<=50K')].shape[0]

greater_percent = np.divide(n_greater_50k, float(n_records)) * 100


print "Total number of records: {}".format(n_records)
print "Individuals making more than $50,000: {}".format(n_greater_50k)
print "Individuals making at most $50,000: {}".format(n_at_most_50k)
print "Percentage of individuals making more than $50,000: {:.2f}%".format(greater_percent)

income_raw = data['income']
features_raw = data.drop('income', axis = 1)
vs.distribution(features_raw)

skewd = ['capital-gain', 'capital-loss']
features_raw[skewd] = data[skewd].apply(lambda x:np.log(x+1))
vs.distribution(features_raw,transformed=True)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
features_raw[numerical] = scaler.fit_transform(data[numerical])

display(features_raw.head(n = 1))
features = pd.get_dummies(features_raw)
income = income_raw.replace(['<=50K', '>50K'], [0, 1])
encoded = list(features.columns)
print "{} total features after one-hot encoding.".format(len(encoded))

from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(features, income, test_size = 0.2, random_state = 0,
                                                    stratify = income)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0,
                                                    stratify = y_train)


print "Training set has {} samples.".format(X_train.shape[0])
print "Validation set has {} samples.".format(X_val.shape[0])
print "Testing set has {} samples.".format(X_test.shape[0])



accuracy = np.divide(n_greater_50k, float(n_records))


recall = np.divide(n_greater_50k, n_greater_50k)
precision = np.divide(n_greater_50k, float(n_records))
fscore = (1 + np.power(0.5, 2)) * np.multiply(precision, recall) / (np.power(0.5, 2) * precision + recall)

print "Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(accuracy, fscore)



from sklearn.metrics import fbeta_score, accuracy_score

def train_predict(learner, sample_size, X_train, y_train, X_test, y_test): 
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''
    
    results = {}
    
 
    start = time()
    learner = learner.fit(X_train[: sample_size], y_train[: sample_size])
    end = time() 
    

    results['train_time'] = end - start
    
 
    start = time() 
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[: 300])
    end = time() 
    
    results['pred_time'] = end - start
            
  
    results['acc_train'] = accuracy_score(y_train[: 300], predictions_train)
        
   
    results['acc_val'] = accuracy_score(y_test, predictions_test)
    

    results['f_train'] = fbeta_score(y_train[: 300], predictions_train, beta=0.5)
        

    results['f_val'] = fbeta_score(y_test, predictions_test, beta=0.5)
       
    
    print "{} trained on {} samples.".format(learner.__class__.__name__, sample_size)
        
    
    return results


from sklearn import tree, svm, ensemble
from sklearn.neighbors import KNeighborsClassifier

clf_A = tree.DecisionTreeClassifier()
clf_B = svm.SVC()
clf_C = ensemble. RandomForestClassifier()


samples_1 = int(X_train.shape[0] * 0.01)
samples_10 = int(X_train.shape[0] * 0.1)
samples_100 = int(X_train.shape[0] * 1)
print [samples_1, samples_10, samples_100]

results = {}
for clf in [clf_A, clf_B, clf_C]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        results[clf_name][i] = train_predict(clf, samples, X_train, y_train, X_test, y_test)

vs.evaluate(results, accuracy, fscore)

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import make_scorer
from sklearn.ensemble import AdaBoostClassifier

clf = AdaBoostClassifier(random_state=0)


parameters = {'n_estimators': [50, 100, 200]}
scorer = make_scorer(fbeta_score, beta=0.5)


kfold = KFold(n_splits=10)
grid_obj = GridSearchCV(clf, parameters, scorer, cv=kfold)

grid_fit = grid_obj.fit(X_train, y_train)

best_clf = grid_obj.best_estimator_


predictions = (clf.fit(X_train, y_train)).predict(X_val)
best_predictions = best_clf.predict(X_val)


print "Unoptimized model\n------"
print "Accuracy score on validation data: {:.4f}".format(accuracy_score(y_val, predictions))
print "F-score on validation data: {:.4f}".format(fbeta_score(y_val, predictions, beta = 0.5))
print "\nOptimized Model\n------"
print "Final accuracy score on the validation data: {:.4f}".format(accuracy_score(y_val, best_predictions))
print "Final F-score on the validation data: {:.4f}".format(fbeta_score(y_val, best_predictions, beta = 0.5))



from sklearn.ensemble import RandomForestClassifier


model = RandomForestClassifier(random_state=0)
model.fit(X_train, y_train)


importances = model.feature_importances_
importances_AdaBoost = best_clf.feature_importances_

vs.feature_plot(importances, X_train, y_train)
vs.feature_plot(importances_AdaBoost, X_train, y_train)


from sklearn.base import clone


X_train_reduced = X_train[X_train.columns.values[(np.argsort(importances)[::-1])[:5]]]
X_val_reduced = X_val[X_val.columns.values[(np.argsort(importances)[::-1])[:5]]]


clf_on_reduced = (clone(best_clf)).fit(X_train_reduced, y_train)


reduced_predictions = clf_on_reduced.predict(X_val_reduced)


print "Final Model trained on full data\n------"
print "Accuracy on validation data: {:.4f}".format(accuracy_score(y_val, best_predictions))
print "F-score on validation data: {:.4f}".format(fbeta_score(y_val, best_predictions, beta = 0.5))
print "\nFinal Model trained on reduced data\n------"
print "Accuracy on validation data: {:.4f}".format(accuracy_score(y_val, reduced_predictions))
print "F-score on validation data: {:.4f}".format(fbeta_score(y_val, reduced_predictions, beta = 0.5))