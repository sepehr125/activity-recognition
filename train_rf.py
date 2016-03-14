# -*- coding: utf-8 -*-

import argparse
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import pickle
from settings import VALID_TARGETS, DATA_COLS, TARGET_COL


"""
This script:
1. Loads given pickled dataframe. 
2. Uses all other columns to predict "target" column with random forest
3. Saves model
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

    args = parser.parse_args()

    #-----------------------------

    # load feature matrix as dataframe:
    df = pickle.load(open(args.feature_matrix, 'rb'))

    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL[0]]
<<<<<<< HEAD

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    model = RandomForestClassifier(
        class_weight='subsample', # takes care of unbalanced classes
        n_estimators=500)
=======

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestClassifier(
        class_weight='subsample', # takes care of unbalanced classes
        n_estimators=200)
>>>>>>> 97e98fbbdee8d0177643b069bcee86a5fd5021e9

    model.fit(X_train, y_train)
    
    # Print overall accuracy
    test_score = model.score(X_test, y_test)
    print("Score on test set: %f"%test_score)

<<<<<<< HEAD
    # compare accuracy to baselines:
    # Baseline 1: a random classifier:
    print("vs. %f if randomly picked"%(1./len(VALID_TARGETS)))

    # Baseline 2: the "average" classifier:
=======
    # compare accuracy to baseline
    print("vs. %f if randomly picked"%(1./len(VALID_TARGETS)))
>>>>>>> 97e98fbbdee8d0177643b069bcee86a5fd5021e9
    popular = max([len(grp) for _, grp in df.groupby('target')]) / len(df)
    print("vs. %f if picked most popular"%popular)

    # Print detailed confusion matrix
    y_pred = model.predict(X_test)
    conf_mat = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(conf_mat)

    # Print features in order of importance 
    features = zip(model.feature_importances_, X.columns)
    features_sorted = sorted(features, key=lambda t: t[0])
    for importance, feature in features_sorted[::-1]:
        print("%f : %s"%(importance, feature))


    # save the model
    with open(args.output_file, 'wb') as f:
        pickle.dump(model, f)

