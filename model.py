from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import joblib

def load_train_data():
    train_data = pd.read_csv('C:/Users/alison/Desktop/flask/train.csv', header=0)

    X_train = train_data.drop(columns='fake')
    y_train = train_data['fake']

    return X_train, y_train

def load_test_data():
    test_data = pd.read_csv('C:/Users/alison/Desktop/flask/test.csv', header=0)

    X_test = test_data.drop(columns='fake')
    y_test = test_data['fake']

    return X_test, y_test

def get_classifier_cv_score(model, X, y, scoring='accuracy', cv=7):
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score=True)
    train_scores = scores['train_score']
    val_scores = scores['test_score']
    
    train_mean = np.mean(train_scores)
    val_mean = np.mean(val_scores)
    
    return train_mean, val_mean

def plot_confusion_matrix(y_actual, y_pred, labels, title=''):
    data = confusion_matrix(y_actual, y_pred)
    ax = sns.heatmap(data,
                     annot=True,
                     cbar=False,
                     fmt='d',
                     xticklabels = labels,
                     yticklabels = labels)
    ax.set_title(title)
    ax.set_xlabel("predicted values")
    ax.set_ylabel("actual values")

def plot_roc_curve(y_test, y_prob):
    y_test = y_test.tolist()
    y_prob = y_prob.tolist()
    tmp_y_prob = []
    tmp_y_prob = sorted(y_prob, key=float, reverse=True)
    count_1 = 0
    count_0 = 0
    TP = 0
    FP = 0
    tpr = []
    fpr = []
    sort_y_prob = []
    for i in range(0, len(y_prob), 1):
        sort_y_prob.append(y_test[y_prob.index(tmp_y_prob[i])])
    count_1 = sort_y_prob.count(1)
    count_0 = sort_y_prob.count(0)

    for i in range(len(sort_y_prob)):
        if sort_y_prob[i] == 1:
            TP += 1
            tpr.append(TP/count_1)
            fpr.append(FP/count_0)
        else :
            FP += 1
            tpr.append(TP/count_1)
            fpr.append(FP/count_0)
    
    plt.plot(fpr, tpr)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig("roc_curve.png")

X_data, y_data = load_train_data()

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=37)

model_list = [LogisticRegression(max_iter=600),
              SVC(), 
              RandomForestClassifier(),]

train_scores = []
val_scores = []

for model in model_list:
    train, val = get_classifier_cv_score(model, X_train, y_train,'average_precision')
    train_scores.append(train)
    val_scores.append(val)
    
models_score = sorted(list(zip(val_scores, train_scores, model_list)), reverse=True)

print("-------------------------------------")
for val, train, model in models_score:
    print("Model: {} ".format(model.__class__.__name__))

    print("train_score: {:.3f}".format(train)) 

    print("validation_score: {:.3f}".format(val)) 

    print("-------------------------------------")

model = RandomForestClassifier()

model_params = {
    'n_estimators': randint(120, 700),
    'max_features': randint(1, 7),
}

random = RandomizedSearchCV(model, model_params, n_iter=100, cv=7, random_state=1)
result = random.fit(X_train, y_train)
# print('Best Score: %s' % result.best_score_)
# print('Best Hyperparameters: %s' % result.best_params_)

pipeline = Pipeline([('preprocessing', StandardScaler()), ('classifier', random.best_estimator_)])
pipeline.fit(X_train, y_train)
joblib.dump(pipeline, 'Random_model')

X_final, y_final = load_test_data()
# print("Test score: {:.3f}".format(pipeline.score(X_final, y_final)))

y_pred = pipeline.predict(X_final)

print("accuracy: %f" % (metrics.accuracy_score(y_test, y_pred > .5)))
print("precision: %f" % (metrics.precision_score(y_test, y_pred > .5)))
print("recall: %f" % (metrics.recall_score(y_test, y_pred > .5)))
plot_roc_curve(y_test.flatten(), y_pred.flatten())

labels = ["genuine", "fake"]
title = "Predicting Fake Instagram Account"
plot_confusion_matrix(y_final, y_pred, labels, title)
