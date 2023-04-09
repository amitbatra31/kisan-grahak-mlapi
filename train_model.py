from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('data/Crop_recommendation.csv')


y=df['label']
X=df.drop('label',axis=1)

# splitting the data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=0)


clf=RandomForestClassifier(n_estimators=20,max_depth=5,bootstrap=False,random_state=0)
clf.fit(X_train,y_train)


y_pred = clf.predict(X_test)

from sklearn import metrics

# Model Accuracy
print("Accuracy:", metrics.accuracy_score(y_pred, y_test))

from sklearn.model_selection import cross_val_score
print(cross_val_score(clf,X,y,cv=10).mean())

from sklearn import preprocessing

# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()

# Encode labels in column 'species'.
df['label_encoded'] = label_encoder.fit_transform(df['label'])

# Make prediction on the test set
test_input = {
     "N": 90,
     "P": 42,
     "K": 43,
     "temperature": 20,
     "humidity": 82,
     "ph":6.4,
     "rainfall":202
    }

# test_input = pd.DataFrame(test_input, index=[0])

# y_predict = clf.predict(X_test)
# y_predict = clf.predict(test_input)

# print(y_predict)

t = pd.DataFrame(test_input, index=[0])
y_pred_prob = clf.predict(t)
print(y_pred_prob)
# y_pred = np.argmax(y_pred_prob, axis=1)
#
# y_pred_class = label_encoder.inverse_transform(y_pred)

# print(y_pred_class)
# Save model
with open('model2.pickle', 'wb') as f:
    pickle.dump(clf, f)
