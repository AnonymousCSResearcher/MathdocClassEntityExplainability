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

from lime.lime_text import LimeTextExplainer

# Set file paths
basePath = 'D:\\NTCIR-12_MathIR_arXiv_Corpus\\'
inputPath = basePath + 'output_Explainability\\'
outputPath = inputPath

# Set mode
mode = "text"

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
with open(inputPath + "entities_labs.pkl", 'rb') as f:
    docLabs = pickle.load(f)
# load texts
with open(inputPath + "entities_text_raw.pkl", 'rb') as f:
    docTexts = pickle.load(f)
# load vectors
#with open(inputPath + "entities_text_tfidf.pkl", 'rb') as f:
#    docVecs = pickle.load(f)

# make pipeline
vectorizer = TfidfVectorizer()
docVecs = vectorizer.fit_transform(docTexts)
pipeline = make_pipeline(vectorizer,classifier)

# fit model
classifier.fit(docVecs,docLabs)

# classification explaination
#https://marcotcr.github.io/lime/tutorials/Lime%20-%20multiclass.html

example_nr = 1
example_text = docTexts[example_nr]
example_lab = docLabs[example_nr]

classes = classifier.classes_

explainer = LimeTextExplainer(class_names=classes)

# create explanations dict for entities
explanations_dict_entities = {}
for example_nr in range(len(docTexts)):
    print("Processing example nr "
          + str(example_nr) + "/" + str(len(docTexts)) + " ...")
    example_text = docTexts[example_nr]
    example_lab = docLabs[example_nr]
    explanations = explainer.explain_instance(example_text,
                                             pipeline.predict_proba)#,
                                             #num_features=3,#len(example_text.split()),
                                             # affects numbers
                                             #top_labels=len(classes))
    for label in explanations.as_list():
        if True: #label[1] > 0:
            try:
                explanations_dict_entities[example_lab][label[0]] += 1
            except:
                try:
                    explanations_dict_entities[example_lab][label[0]] = 1
                except:
                    explanations_dict_entities[example_lab] = {}
                    explanations_dict_entities[example_lab][label[0]] = 1

# # create explanations dict for classes
# explanations_dict_classes = {}
# for class_nr in range(len(classes)):
#     explanations_dict_classes[classes[class_nr]]\
#         = explanations.as_list(class_nr)

# save explanations dict for entities
with open(outputPath + "most_discriminative_" + mode + "_class_entities.json","w",
          encoding='utf8') as f:
    json.dump(explanations_dict_entities,f)

# # print prediction and explanation
# class_nr = 5
# print("True label: " + example_lab)
# print("Confidence: " + str(pipeline.predict_proba([example_text])))
# print("Explanation for class '" + docLabs[class_nr] + "': "
#       + str(explanations.as_list(label=class_nr)))
# #print([docLabs[index] for index in explanations.available_labels()])
# #explanations.show_in_notebook(text=False)

print("end")