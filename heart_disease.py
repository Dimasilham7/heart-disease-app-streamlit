import pandas as pd
heart = pd.read_csv('heart.csv')

df = heart.copy()
target = 'sex'
encode = ['fbs','exang']

for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]

target_mapper = {'female':0, 'male':1}
def target_encode(val):
    return target_mapper[val]

df['sex'] = df['sex'].apply(target_encode)

# Separating X and y
X = df.drop('sex', axis=1)
Y = df['sex']

# Build random forest model
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X, Y)

# Saving the model
import pickle
pickle.dump(clf, open('heart.pkl', 'wb'))