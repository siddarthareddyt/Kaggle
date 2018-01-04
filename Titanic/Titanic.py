#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 22:45:49 2017

@author: siddartha
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.feature_selection
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import minmax_scale
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import ShuffleSplit
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn import tree

train = pd.read_csv("train.csv")
holdout = pd.read_csv("test.csv")
print(holdout.head(n=10))

train.describe()
holdout.describe()
train.isnull().sum()

#Data Exploration

#Sex
import matplotlib.pyplot as plt

sex_pivot = train.pivot_table(index="Sex",values="Survived")
sex_pivot.plot.bar()
plt.show()

#Pclass
plcass_pivot = train.pivot_table(index="Pclass", values="Survived")
plcass_pivot.plot.bar()
plt.show()

#SibSp, Parch
explore_cols = ["SibSp","Parch","Survived"]
explore = train[explore_cols].copy()
explore['familysize'] = explore[["SibSp","Parch"]].sum(axis=1)
pivot = explore.pivot_table(index="familysize",values="Survived")
pivot.plot.bar(ylim=(0,1),yticks=np.arange(0,1,.1))
plt.show()


#Engineer new features
train = preprocess(train)
holdout = preprocess(holdout)

#after age feature engineered
age_cat_pivot = train.pivot_table(index="Age_categories", values="Survived")
age_cat_pivot.plot.bar()
plt.show()


#feature Selection using RFECV
columns = ['Age_categories_Missing', 'Age_categories_Infant',
       'Age_categories_Child', 'Age_categories_Teenager',
       'Age_categories_Young Adult', 'Age_categories_Adult',
       'Age_categories_Senior', 'Pclass_1', 'Pclass_2', 'Pclass_3',
       'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S',
       'Fare_categories_0-12',
       'Fare_categories_12-50','Fare_categories_50-100', 'Fare_categories_100+',
       'Title_Master', 'Title_Miss', 'Title_Mr','Title_Mrs', 'Title_Officer',
       'Title_Royalty', 'Cabin_type_A','Cabin_type_B', 'Cabin_type_C', 'Cabin_type_D',
       'Cabin_type_E','Cabin_type_F', 'Cabin_type_G', 'Cabin_type_Unknown', 'isalone']

optimized_features = select_features(train, columns)

ann = ann_model(train, optimized_features)
y_pred = ann.predict(np.array(holdout[optimized_features]))

predictions =pd.DataFrame(0, index=np.arange(len(y_pred)), columns=["pred"])
y_pred = y_pred[:,0]
predictions["rawpred"] = y_pred
predictions["pred"].loc[predictions['rawpred'] >= 0.5] = 1
submission_df = {"PassengerId": holdout["PassengerId"], "Survived": predictions["pred"]}
submission = pd.DataFrame(submission_df)
submission.to_csv(path_or_buf="SubmissionAnn.csv", index=False, header=True)

save_submission(ann, columns, "SubmissionAnn.csv")


#model selection using Grid Search
models = select_model(train, columns)
best_grid = models[0]["grid"]
best_classifier = models[0]["model"]
best_params = models[0]["best_params"]


scores = cross_val_score(best_classifier, train[columns], train["Survived"], cv=10)
accuracy = scores.mean()

holdout_predictions = best_classifier.predict(holdout[columns])

#holdout prediction
save_submission(best_classifier, columns, "SubmissionDum.csv")


def ann_model(df, features):

    
    # Initialising the ANN
    classifier = Sequential()
    # Adding the input layer and the first hidden layer
    classifier.add(Dense(input_dim=len(features), units=15, activation="relu", kernel_initializer="uniform"))
    

    # Adding the second hidden layer
    classifier.add(Dense(units = 15, kernel_initializer = 'uniform', activation = 'relu'))
    
    #classifier.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'relu'))

    # Adding the output layer
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    
    # Compiling the ANN
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    # Fitting the ANN to the Training set
    train_X = df[features]
    train_y = df["Survived"]
    classifier.fit(np.array(train_X), np.array(train_y), batch_size = 10, epochs = 100)
    
    
    return classifier

#pre-processing functions

def process_missing(df):
    df["Fare"] = df["Fare"].fillna(train["Fare"].mean())
    df["Embarked"] = df["Embarked"].fillna("S")
    return df

def process_age(df):
    df["Age"] = df["Age"].fillna(-0.5)
    cut_points = [-1,0,5,12,18,35,60,100]
    label_names = ["Missing","Infant","Child","Teenager","Young Adult","Adult","Senior"]
    df["Age_categories"] = pd.cut(df["Age"],cut_points,labels=label_names)
    return df

def process_fare(df):
    cut_points = [-1,12,50,100,1000]
    label_names = ["0-12","12-50","50-100","100+"]
    df["Fare_categories"] = pd.cut(df["Fare"],cut_points,labels=label_names)
    return df

def process_cabin(df):
    df["Cabin_type"] = df["Cabin"].str[0]
    df["Cabin_type"] = df["Cabin_type"].fillna("Unknown")
    df = df.drop('Cabin',axis=1)
    return df    


def process_titles(df):
    titles = {
        "Mr" :         "Mr",
        "Mme":         "Mrs",
        "Ms":          "Mrs",
        "Mrs" :        "Mrs",
        "Master" :     "Master",
        "Mlle":        "Miss",
        "Miss" :       "Miss",
        "Capt":        "Officer",
        "Col":         "Officer",
        "Major":       "Officer",
        "Dr":          "Officer",
        "Rev":         "Officer",
        "Jonkheer":    "Royalty",
        "Don":         "Royalty",
        "Sir" :        "Royalty",
        "Countess":    "Royalty",
        "Dona":        "Royalty",
        "Lady" :       "Royalty"
    }
    extracted_titles = df["Name"].str.extract(' ([A-Za-z]+)\.',expand=False)
    df["Title"] = extracted_titles.map(titles)
    return df

def process_family(df):
    numFam = df[["SibSp","Parch"]].sum(axis=1)
    df["isalone"] = np.where(numFam>=1, 1, 0)
    return df

def create_dummies(df,column_name):
    dummies = pd.get_dummies(df[column_name],prefix=column_name)
    df = pd.concat([df,dummies],axis=1)
    return df

def plot_correlation_heatmap(df):
    corr = df.corr()
    
    sns.set(style="white")
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)


    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.show()

def preprocess(df):
    df = process_missing(df)
    df = process_age(df)
    df = process_titles(df)
    df = process_cabin(df)
    df = process_fare(df)
    df = process_family(df)

    
    columnNames = ["Age_categories", "Pclass", "Sex", "Fare_categories", "Title", "Cabin_type", "Embarked"]
    for column in columnNames:
        df = create_dummies(df, column)
    return df

def select_features(df, columns, model=None):
    newDf = df.copy()
    newDf = newDf.select_dtypes(['number'])
    newDf = newDf.dropna(axis=1, how='any')
    
    #dropColumns = ["PassengerId", "Survived"]
    #newDf = newDf.drop(dropColumns, axis = 1)
    
    all_X = newDf[columns]
    all_y = df["Survived"]
    
    cv = StratifiedShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0)
    if model == None:
        classifier = tree.DecisionTreeClassifier(
                criterion="entropy",
                max_depth=10,
                max_features='auto',
                max_leaf_nodes=None,
                min_impurity_decrease = 0.0,
                min_samples_leaf = 10,
                min_samples_split = 3
                )
    else:
        classifier = model
    selector = RFECV(classifier, scoring = 'roc_auc', cv=cv, step = 1)
    selector.fit(all_X,all_y)
    optimized_columns = all_X.columns[selector.support_]
    return optimized_columns

def select_model(df, features):
    
    train_X = df[features]
    train_y = df["Survived"]
    
    
    cv = ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0)
    
    """x= {
                    "name": "RandomForestClassifier",
                    "estimator": RandomForestClassifier(random_state=0),
                    "hyperparameters":
                        {
                                "n_estimators": [20,25, 35, 40,45, 50, 55, 60, 65, 70, 75],
                                "criterion": ["entropy", "gini"],
                                "max_features": ["log2", "sqrt"],
                                "min_samples_leaf": [1, 5, 8],
                                "min_samples_split": [2, 3, 5]
                        }
            }"""
    model_params = [
            {
                    "name": "DecisionTreeClassifier",
                    "estimator": tree.DecisionTreeClassifier(),
                    "hyperparameters":
                        {
                                "criterion": ["entropy", "gini"],
                                "max_depth": [None, 2,4,6,8,10, 12, 14, 16],
                                'min_samples_split': [2,3,4,5,10,.03,.05,.1],
                                "max_features": [None, "auto"],
                                "min_samples_leaf": [1,2,3,4,5,10,12, .5, .03,.05,.1]
                        }
            },
            
            {
                    "name": "KernelSVMClassifier",
                    "estimator": SVC(random_state=0),
                    "hyperparameters":
                        {
                                "kernel": ["rbf"],
                                "C": np.logspace(-9, 3, 13),
                                "gamma": np.logspace(-9, 3, 13)
                        }
            } ,
            {
                    "name": "KNeighborsClassifier",
                    "estimator": KNeighborsClassifier(),
                    "hyperparameters":
                        {
                                "n_neighbors": range(1,20,2),
                                "weights": ["distance", "uniform"],
                                "algorithm": ["ball_tree", "kd_tree", "brute"],
                                "p": [1,2]
                        }
            },
            {
                    "name": "LogisticRegressionClassifier",
                    "estimator": LogisticRegression(),
                    "hyperparameters":
                        {
                                "solver": ["newton-cg", "lbfgs", "liblinear"]
                        }
            }          
            ]
    models = []
    for model in model_params:
        print(model["name"])
        grid = GridSearchCV(model["estimator"], 
                            param_grid=model["hyperparameters"], 
                            cv=10)
        grid.fit(train_X, train_y)
        
        model_att = {
                "model": grid.best_estimator_, 
                "best_params": grid.best_params_, 
                "best_score": grid.best_score_,
                "grid": grid
                }
        models.append(model_att)
        print("Evaluated model and its params: ")
        print(grid.best_params_)
        print(grid.best_score_)
    return models

def save_submission(classifier, features, filename, isann=False):
    holdout_predictions = classifier.predict(holdout[features])
    if isann:
        holdout_predictions = (holdout_predictions > 0.5)
    submission_df = {"PassengerId": holdout["PassengerId"], "Survived": holdout_predictions}
    submission = pd.DataFrame(submission_df)
    submission.to_csv(path_or_buf=filename, index=False, header=True)
    
        


