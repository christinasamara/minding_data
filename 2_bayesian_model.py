import pandas as pd
from sklearn.model_selection import train_test_split
from pgmpy.models import BayesianModel
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.estimators import HillClimbSearch, BDeuScore
import glob
from sklearn.metrics import accuracy_score

def read_dataset():
    li = []
    for filename in glob.glob("harth\*"):
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)

    frame = pd.concat(li, axis=0, ignore_index=True)
    frame = frame.drop(['index', 'Unnamed: 0'], axis=1)

    frame = frame.drop(['timestamp'], axis=1)


    return frame

data=read_dataset()

num_bins = 20 
for column in data.select_dtypes(include=['float64']):
    data[column] = pd.qcut(data[column], q=num_bins, labels=False, duplicates='drop')

print(data.head)

X = data  
y = data['label']  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = BayesianNetwork()

hc = HillClimbSearch(X_train)
best_model = hc.estimate(scoring_method=BDeuScore(X_train))
model.add_edges_from(best_model.edges())


model.fit(X_train, estimator=BayesianEstimator, prior_type="BDeu", equivalent_sample_size=10)
y_pred_train = model.predict(X_test.drop(columns=['label']))


y_pred_train.to_csv('y_predictions_bayesian.csv', index=False)
y_test.to_csv('y_test_bayesian.csv', index=False)
