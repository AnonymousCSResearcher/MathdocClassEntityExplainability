import pickle
import numpy as np

import os
import time

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

# Set file paths
#basePath = ""
basePath = 'D:\\NTCIR-12_MathIR_arXiv_Corpus\\'

inputPath = basePath + 'output_Explainability\\'

# initialize classifier

# single classifiers
# classifier = LogisticRegression()
classifier = LinearSVC()
# classifier = SVC() # Radial basis function kernel
# (
# classifier = GaussianNB()
# classifier = MultinomialNB()
# )
# classifier = DecisionTreeClassifier()
# classifier = MLPClassifier(hidden_layer_sizes=(500, )) # hidden_layer_sizes=(500, )
# classifier = KNeighborsClassifier()

# ensemble classifiers
# classifier = GradientBoostingClassifier()
# classifier = RandomForestClassifier()

# evaluation functions

def classification_score(docVecs,docLabs,classifier):

    result = np.mean(cross_val_score(classifier, docVecs, docLabs, cv=10))
    return np.round(np.multiply(result,100),1)

# load labels
with open(inputPath + "entities_labs.pkl", 'rb') as f:
    docLabs = pickle.load(f)
# load vectors
with open(inputPath + "entities_math_tfidf.pkl", 'rb') as f:
    docVecs = pickle.load(f)

# classification evaluation

# start timer
t0 = time.time()

accuracy = classification_score(docVecs,docLabs,classifier)

# stop timer
t1 = time.time()

# time elapsed and result
duration = str(t1 - t0)
print("Duration: " + str(duration))
print("Accuracy: " + str(accuracy))

print("end")