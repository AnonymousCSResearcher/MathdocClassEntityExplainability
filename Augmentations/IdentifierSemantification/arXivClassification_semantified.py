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
basePath = "E:\\NTCIR-12_MathIR_arXiv_Corpus\\"

inputPath = basePath + "ML_output_balanced\\arXivEmbeddings\\"
outputPath = basePath + "ML_output_balanced\\arXivClassification\\"
# inputPath = basePath + "ML_output_balanced\\arXivEmbeddings_abstract\\"
# outputPath = basePath + "ML_output_balanced\\arXivClassification_abstract\\"

# initialize classifier
#classifier_name = "LogisticRegression"

# single classifiers
classifier = LogisticRegression()
# classifier = LinearSVC()
# classifier = SVC() # Radial basis function kernel
# (
# classifier = GaussianNB()
# classifier = MultinomialNB()
# )
# classifier = DecisionTreeClassifier()
# classifier = MLPClassifier()#hidden_layer_sizes=(500, )) # hidden_layer_sizes=(500, )
# classifier = KNeighborsClassifier()

# ensemble classifiers
# classifier = GradientBoostingClassifier()
# classifier = RandomForestClassifier()

# evaluation functions

def classification_score(docVecs,docLabs,classifier):

    #classifier.fit(docVecs, docLabs)

    #individual_result = cross_val_score(classifier, docVecs, docLabs, cv=10)

    result = np.mean(cross_val_score(classifier, docVecs, docLabs, cv=10,verbose=True))
    return np.round(np.multiply(result,100),1)

# load labels
with open(inputPath + "secLabs_sixclass.pkl", 'rb') as f:
    docLabs = pickle.load(f)
# with open(inputPath + "docLabsMath_abstract.pkl", 'rb') as f:
#     docLabs = pickle.load(f)

def print_report_score(name):
    # load vectors
    with open(inputPath + name + ".pkl", 'rb') as f:
        docVecs = pickle.load(f)

    # calculate and report score
    score = classification_score(docVecs,docLabs,classifier)
    line = str(name) + ": " + str(score)
    print(line)
    report.append(line)
    print("")
    report.append("")

# classification evaluation

# start timer
t0 = time.time()

report = []

# classification executions
# embeddings = []
# for file in os.listdir(inputPath):
#     embeddings.append(file.strip(".pkl"))
# embeddings.remove("docLabs")
embeddings = ["secText_sixclass_tfidf",
              "secText_sixclass_annomathtex_tfidf2",
              "secText_sixclass_textandcategoryconcept_tfidf",
              "secMath_sixclass_opid_tfidf"]
#embeddings = ["secText_threeclass_tfidf",
#               "secMath_threeclass_opid_tfidf",
#               "secText_threeclass_annomathtex_tfidf",
#               "secMath_threeclass_annomathtex_tfidf"]
#embeddings = ["docTextMath_surroundings_tfidf"]
#embeddings = ["docMath_semantics_Wikipedia_tfidf"]
#embeddings = ["docMath_semantics_Wikidata_tfidf"]
#embeddings = ["docText_tfidf"]
#embeddings = ["doc2vecText","docText_tfidf","doc2vecMath_op","docMath_op_tfidf","doc2vecMath_id","docMath_id_tfidf","doc2vecMath_opid","docMath_opid_tfidf","doc2vecMath_semantics","docMath_semantics_tfidf","doc2vecTextMath_opid","doc2vecTextMath_semantics"]
#embeddings = ["doc2vecText_abstract", "docText_tfidf_abstract"]
#embeddings = ["doc2vecMath_opid_abstract", "docMath_opid_tfidf_abstract"]

for embedding in embeddings:
    try:
        print_report_score(embedding)
    except:
        line = embedding + ": N/A"
        print(line)
        report.append(line)
        print("")
        report.append("")

# stop timer
t1 = time.time()

# time elapsed
duration = str(t1 - t0)
line = "Duration: " + str(duration)
print(line)
report.append(line)

# save report
# with open (outputPath + classifier_name + ".txt", "w") as f:
#     for line in report:
#         f.write(line + "\n")

# (todo: classify formulae instead of documents)
# todo: analyse links (correlation) between arXiv fields

print("end")