import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import json

# load table
filename = "msc_categories.csv"
table = pd.read_csv(filename,sep=r'\s*,\s*')
data = table[['msc','categories']]
del table

# Init x and y label lists
X = []
Y = []

for index in range(0,len(data)):
    mscs_text = data['msc'].iloc[index]
    categories_text = data['categories'].iloc[index]

    # predict categories from mscs
    # predict single-label
    #X.append(mscs_text)
    #Y.append(categories_text.split()[0]) #.split(".")[0])
    # predict multi-label
    #for category in categories_text.split():
    #    X.append(mscs_text)
    #    Y.append(category)#.split(".")[0]) # fine or coarse granular?

    # predict mscs from categories
    # predict single-label
    X.append(categories_text)
    n_MSCdigits = 2
    Y.append(mscs_text.split()[0][0:n_MSCdigits])
    # predict multi-label
    #for msc in mscs_text.split():
    #    X.append(categories_text)
    #    Y.append(msc[0:1]) # fine or coarse granular?

# Generate tf_idf vectors
X_vec = TfidfVectorizer().fit_transform(X)

# Evaluate classification
classifier = LogisticRegression()
classifier.fit(X_vec,Y)
predictions = classifier.predict(X_vec)
#accuracy = np.mean(cross_val_score(classifier, X, Y, cv=10))
#print(accuracy)

# Create predictions dict
predictions_dict = {}
# count
for idx in range(0,len(predictions)):
    for predictor in X[idx].split():
        try:
            predictions_dict[predictor].append(predictions[idx])
        except:
            predictions_dict[predictor] = [predictions[idx]]
# predict
for subject_class in predictions_dict.items():
    predictions_dict[subject_class[0]] = str(np.argmax(np.bincount(subject_class[1])))

# save to json
with open("predictions_classifier.json",'w') as f:
    json.dump(predictions_dict,f)

print("end")