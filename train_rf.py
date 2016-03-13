# -*- coding: utf-8 -*-

import argparse
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import GradientBoostingClassifier # generally performs worse
from sklearn.metrics import confusion_matrix

import pickle


"""
This script:
1. Loads given, pickled dataframe. 
2. Looks for "target" column
3. Uses all other columns to predict target with random forest
4. Saves model
"""

if __name__ == '__main__':


    parser = argparse.ArgumentParser(
                        description=
                        'Trains random forest classifier \
                        on given data')

    parser.add_argument(
                        'feature_matrix',
                        help=
                        "Pickled DataFrame with 'target' column \
                        and at least one feature.")

    parser.add_argument(
                        'output_file',
                        help="Path or name of file to save model as.")

    #TODO: pass in test_size and number of estimators

    args = parser.parse_args()

    #-----------------------------

    # load feature matrix as dataframe:
    df = pickle.load(open(args.feature_matrix, 'rb'))

    if 'target' not in df.columns:
        raise KeyError("Target column not found. Nothing to predict")

    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


    model = RandomForestClassifier(
        class_weight='subsample', 
        n_estimators=100)
    
    # model = GradientBoostingClassifier() # generally performs worse

    model.fit(X_train, y_train)
    
    # Print overall accuracy
    test_score = model.score(X_test, y_test)
    print("Test score: %f"%test_score)
    
    # Print detailed confusion matrix
    y_pred = model.predict(X_test)
    conf_mat = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(conf_mat)

    # Print features in order of importance by tree model 
    features = zip(model.feature_importances_, X.columns)
    features_sorted = sorted(features, key=lambda t: t[0])
    for importance, feature in features_sorted[::-1]:
        print("%f : %s"%(importance, feature))

    # save the model
    with open(args.output_file, 'wb') as f:
        pickle.dump(model, f)

