import csv
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def read_dataset():
    li = []
    for filename in glob.glob("harth/*"):
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)

    frame = pd.concat(li, axis=0, ignore_index=True)
    frame = frame.drop(['index', 'Unnamed: 0'], axis=1)

    frame['timestamp'] = pd.to_datetime(frame['timestamp'])
    frame['time_diff'] = frame['timestamp'].diff().dt.total_seconds()
    frame = frame.drop(['timestamp'], axis=1)
    frame['time_diff'] = frame['time_diff'].fillna(0)

    print(frame.groupby('label').describe())
    frame.info()

    return frame



def run_classification(X, Y):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(max_depth=30, min_samples_leaf=1, min_samples_split=2, n_estimators=50)
    print("Training started")
    model.fit(X_train, y_train)
    print("Performing predictions")
    predictions = model.predict(X_test)
    
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = accuracy_score(y_test, predictions)

    predictions_df = pd.DataFrame(predictions)
    predictions_df.to_csv('y_predictions_random.csv', index=False)
    y_test.to_csv('y_test_random.csv', index=False)


    print("---------------------------------------------------------------------")
    print(f"Random Forest Classifier yields training accuracy of {train_accuracy} with a testing accuracy of {test_accuracy}")

    return model, train_accuracy, test_accuracy


print("Current time:", datetime.now().time())
frame = read_dataset()
X = frame.drop(columns=['label'])
Y = frame['label']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print('Running Random Forest model')
classifier, train_acc, test_acc = run_classification(X, Y)