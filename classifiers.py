from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


feature_columns = ['PER',
                   'TS%',
                   'AST%',
                   'ORB%',
                   'DRB%',
                   'TRB%',
                   'STL%',
                   'BLK%',
                   'USG%',
                   '3PAr'
                ]

class_column = 'Pos'


url = 'https://raw.githubusercontent.com/mmarti34/Data-Science/master/Final%20Project/2016_NBA_advanced.csv'
df = pd.read_csv(url, index_col=0)

df = df[(df['G'] > 1)]

df = df.dropna(0)

df['pos_num'] = df.Pos.map({'PG':1, 'SG':2, 'SF':3, 'PF':4, 'C':5, 'PF-C':45, 'SG-SF':23})

classifiers = [GaussianNB(), KNeighborsClassifier(), DecisionTreeClassifier()]

training_range = np.arange(0.50, 0.90, 0.05)
results = dict((clf, dict((tr, 0) for tr in training_range)) for clf in classifiers)
n_trials = 10

for i in training_range:
    for j in range(n_trials):
        df['is_training'] = np.random.uniform(0, 1, len(df)) <= i
        training_set = df[df['is_training']==True]
        testing_set = df[df['is_training']==False]

        trainingFeatures = training_set[feature_columns]
        trainingTargets = training_set[class_column]

        testingFeatures = testing_set[feature_columns]
        testingTargets = testing_set[class_column]

        for classifier in classifiers:
            print "------------------------------------------"
            print classifier

            classifier.fit(trainingFeatures, trainingTargets)
            results[classifier][i] += classifier.score(testingFeatures, testingTargets)

            predictions = classifier.predict(testingFeatures)
            print metrics.classification_report(testingTargets, predictions)

for classifier in classifiers:
    plt.plot(training_range, [(results[classifier][d]/float(n_trials)) for d in training_range], label=type(classifier))

plt.legend(loc=9, bbox_to_anchor=(0.5, -0.03), ncol=3)
plt.show()


print(predictions + testingTargets)