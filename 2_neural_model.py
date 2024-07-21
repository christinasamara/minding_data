import glob
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def read_dataset():
    li = []
    for filename in glob.glob("harth\*"):
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)

    frame = pd.concat(li, axis=0, ignore_index=True)
    frame = frame.drop(['index', 'Unnamed: 0'], axis=1)

    frame['timestamp'] = pd.to_datetime(df['timestamp'])
    frame['time_diff'] = frame['timestamp'].diff().dt.total_seconds()
    frame = frame.drop(['timestamp'], axis=1)
    frame['time_diff'] = frame['time_diff'].fillna(0)

    print(frame.groupby('label').describe())
    frame.info()
    
    #mean_df = frame.groupby('label').mean(numeric_only=True)
    #stdev_df = frame.groupby('label').std()
    #print(mean_df)
    #print(stdev_df)
    return frame


def run_classification(X,Y):
    X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.2, random_state=42)
    model = MLPClassifier(hidden_layer_sizes=(128, 64, 32, 64), activation='relu', verbose=1, max_iter=10, random_state=42)
    print("training started")
    model.fit(X_train,y_train)
    print("performing predictions")
    y_predictions = model.predict(X_test)
    print("---------------------------------------------------------------------")
    print(f"Classifier Neural Networks yeilds training accuracy of {model.score(X_train,y_train)}\n with a testing accuracy of {accuracy_score(y_test, y_predictions)}")
    
    y_predictions_df = pd.DataFrame(y_predictions, columns=['Predictions'])
    y_predictions_df.to_csv('y_predictions_neural.csv', index=False)
    y_test.to_csv('y_test_neural.csv', index=False)




frame = read_dataset()
X = frame.drop(['label'],axis = 1)
Y = frame['label']

models=["N_Net"]
models_train_acc = []
models_test_acc = []
run_classification(X,Y)