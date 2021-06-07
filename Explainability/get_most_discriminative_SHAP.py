import pickle
import json

from sklearn.linear_model import LogisticRegression
#from sklearn.naive_bayes import MultinomialNB
#from sklearn.naive_bayes import GaussianNB
#from sklearn.svm import LinearSVC
#from sklearn.svm import SVC
#from sklearn.neural_network import MLPClassifier
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.neighbors import KNeighborsClassifier

#from sklearn.ensemble import GradientBoostingClassifier
#from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline

#https://github.com/slundberg/shap
import shap

# Set file paths
basePath = 'D:\\NTCIR-12_MathIR_arXiv_Corpus\\'
inputPath = basePath + 'output_Explainability\\100perClass\\'
outputPath = inputPath

# initialize classifier

# single classifiers
classifier = LogisticRegression()
# classifier = LinearSVC()
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

# load labels
with open(inputPath + "docLabs.pkl", 'rb') as f:
    docLabs = pickle.load(f)
# load texts
with open(inputPath + "docTexts.pkl", 'rb') as f:
    docTexts = pickle.load(f)
# load vectors
#with open(inputPath + "docVecs.pkl", 'rb') as f:
#    docVecs = pickle.load(f)

# make pipeline
vectorizer = TfidfVectorizer()
docVecs = vectorizer.fit_transform(docTexts)
pipeline = make_pipeline(vectorizer,classifier)

# fit model
print("fit...")
classifier.fit(docVecs,docLabs)

# classification explaination
#https://github.com/slundberg/shap

classes = classifier.classes_

# explain the model's predictions using SHAP
print("explain...")
#explainer = shap.Explainer(pipeline)
#shap_values = explainer([docTexts[0]])
explainer = shap.Explainer(classifier,docVecs)
shap_values = explainer(docVecs[0])
#print(shap_values)
print("visualize...")

print("end")