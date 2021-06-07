import pickle
import json
from os import listdir

from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer

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

# https://github.com/marcotcr/lime
from lime.lime_text import LimeTextExplainer

#https://github.com/slundberg/shap

# Set file paths
basePath = 'D:\\NTCIR-12_MathIR_arXiv_Corpus\\'
datasetPath = basePath + "NTCIR12\\"
inputPath = basePath + 'output_Explainability\\'
outputPath = inputPath

# Set mode
mode = "text"

# Iterate corpus

#TODO: experiment with variables word window, and formula concept definition

#valid_folder_prefix = ["0001"]
#valid_folder_prefix = ["00", "01", "02", "03", "04", "05", "06"]
#valid_folder_prefix = ["00", "01"]
valid_folder_prefix = [""]

# Create inverse_semantic_index_formula_catalog
formula_concept_name_candidates = {}

# Define class counter and desired classes
classCounter = {}
classLimit = 100
#desired_classes = ['astro-ph']
#desired_classes = ['astro-ph','gr-qc']
desired_classes = ['astro-ph', 'cond-mat', 'gr-qc', 'hep-lat', 'hep-ph', 'hep-th', 'math-ph', 'nlin', 'quant-ph', 'physics']

# retrieve text data from document
tokenizer = RegexpTokenizer(r'\w+')
stopword_set = set(stopwords.words('english'))
def get_text(Dir,File):

    raw_str = open(datasetPath + "\\" + Dir + "\\" + File, "r",
                   encoding="utf8").read()

    lowered_str = raw_str.lower()
    tokenized_str = tokenizer.tokenize(lowered_str)
    swremoved = list(set(tokenized_str).difference(stopword_set))
    cleaned = "" # word list for doc2vec
    for word in swremoved:
        contains_digit = False
        for char in word:
            if char.isdigit():
                contains_digit = True
        if not contains_digit and len(word) > 3:
            cleaned += word + " " # text word string for tfidf
    #print(cleaned)
    return cleaned

# exclude formulae, stopwords, html and letters from candidates
excluded = [">", "<", "=", "~",'"', "_"]
with open("../../stopwords.txt") as f:
    stopwords = [line.strip() for line in f]
#invalid = ["times"]
with open("../../letters.txt") as f:
    letters = [line.strip() for line in f]

def findall(p, s):
    '''Yields all the positions of
    the pattern p in the string s.'''
    i = s.find(p)
    while i != -1:
        yield i
        i = s.find(p, i + 1)

# retrieve math data (formulae) from document
def get_math(Dir,File):
    docMath = ""
    with open(datasetPath + "\\" + Dir + "\\" + File, "r", encoding="utf8") as f:
        filestring = f.read()
        formulae = BeautifulSoup(filestring, 'html.parser').find_all('formula')

    # augment formula concept name candidate catalog
    formula_concept_name_candidates[File] = {}
    for formula in formulae:

        # formulaString = str(formula.contents)
        #
        # # extract TeX formula
        # # formulaString
        # s = str(formula.contents)
        # start = 'alttext="'
        # end = '" display='
        try:
            # TeX = re.search('%s(.*)%s' % (start, end), s).group(1)
            TeX = formula.contents[0].attrs['alttext']
        except:
            TeX = ""

        # extract surrounding tex
        index = filestring.find('alttext="' + TeX + '" display=')
        surrounding_text_candidates = filestring[index - 500:index + 500]

        for word in surrounding_text_candidates.split():
            # lowercase and remove .,-()
            word = word.lower()
            char_excl = [".", ":", ",", "-", "(", ")", '=']
            for c in char_excl:
                word = word.replace(c, "")
            # not part of a formula environment
            not_formula = not True in [ex in word for ex in excluded]
            # not stopword
            not_stopword = word not in stopwords
            # not invalid html
            # not_invalid = not True in [inv in word for inv in invalid]
            # not a latin or greek letter
            not_letter = word not in letters
            if not_formula and not_stopword and not_letter:  # and not_invalid
                # if TeX != "" and TeX not in stopwords and TeX not in letters:
                # check if around equation
                # if '=' in TeX:
                if len(TeX) > 10:
                    docMath += word + " "
                    try:
                        formula_concept_name_candidates[File][word][TeX] += 1
                    except:
                        try:
                            formula_concept_name_candidates[File][word][TeX] = 1
                        except:
                            formula_concept_name_candidates[File][word] = {}
                            formula_concept_name_candidates[File][word][TeX] = 1
    return docMath

# Collect doc text and labs for classification
docTexts = []
docLabs = []

# Fetch content and labels of documents
for Dir in listdir(datasetPath):
    for prefix in valid_folder_prefix:
        if Dir.startswith(prefix):
            for File in listdir(datasetPath + "\\" + Dir):
                if not File.startswith("1") and File.endswith(".tei"):
                    # fetch label from file prefix
                    if Dir.startswith("9"):
                        classLab = File.split("9")[0]
                    else:
                        classLab = File.split("0")[0]
                    # check if class is desired and limit is not exceeded
                    try:
                        classCounter[classLab] += 1
                    except:
                        classCounter[classLab] = 1
                    #if True: # switch off desired_classes / classLimit constraints
                    if classLab in desired_classes and classCounter[classLab] <= classLimit:
                        print(Dir + "\\" + File)

                        # Augment doc LABS, TEXT/MATH

                        # LABS
                        docLabs.append(classLab)

                        # TEXT
                        if mode == "text":
                            docText = get_text(Dir,File)
                            docTexts.append(docText[:-1])

                        # MATH
                        elif mode == "math":
                            docMath = get_math(Dir,File)
                            docTexts.append(docMath[:-1])

                        # TEXT & MATH

                        # add
                        # Concatenate docText and docMath
                        #docTexts.append(docText[:-1] + " " + docMath[:-1]) # cut off last space

                        # subtract
                        # Calculate set difference
                        #diff = set(docText.split()) - set(docMath.split())
                        #text = ""
                        #for word in diff:
                        #    text += word + " "
                        #docTexts.append(text[:-1]) # cut off last space

# generate tf-idf docVecs for formula concept docTexts
vectorizer = TfidfVectorizer()
docVecs = vectorizer.fit_transform(docTexts)

# save entities docTexts, docVecs, and docLabs for classification
with open(outputPath + "docTexts.pkl","wb") as f:
    pickle.dump(docTexts,f)
with open(outputPath + "docVecs.pkl","wb") as f:
    pickle.dump(docVecs,f)
with open(outputPath + "docLabs.pkl","wb") as f:
    pickle.dump(docLabs,f)

# save formula_concept_name_candidates

#with open(outputPath + 'formula_concept_name_candidates.json','w',encoding='utf8') as f:
#    json.dump(formula_concept_name_candidates,f)

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
#with open(inputPath + "docLabs.pkl", 'rb') as f:
#    docLabs = pickle.load(f)
# load texts
#with open(inputPath + "docTexts.pkl", 'rb') as f:
#    docTexts = pickle.load(f)
# load vectors
#with open(inputPath + "docVecs.pkl", 'rb') as f:
#    docVecs = pickle.load(f)

# make pipeline
#vectorizer = TfidfVectorizer()
#docVecs = vectorizer.fit_transform(docTexts)
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

# test
#explainer.explain_instance(docTexts[?],pipeline.predict_proba).as_list()

# create explanations dict for entities
explanations_dict_class_entities = {}
for example_nr in range(len(docTexts)):
    print("Processing example nr "
          + str(example_nr) + "/" + str(len(docTexts)) + " ...")
    example_text = docTexts[example_nr]
    example_lab = docLabs[example_nr]
    try:
        explanations = explainer.explain_instance(example_text,
                                                 pipeline.predict_proba)#,
                                                 #num_features=3,#len(example_text.split()),
                                                 # affects numbers
                                                 #top_labels=len(classes))
        for label in explanations.as_list():
            if True: #label[1] > 0:
                try:
                    explanations_dict_class_entities[example_lab][label[0]] += 1
                except:
                    try:
                        explanations_dict_class_entities[example_lab][label[0]] = 1
                    except:
                        explanations_dict_class_entities[example_lab] = {}
                        explanations_dict_class_entities[example_lab][label[0]] = 1#
    except:
        pass

# # create explanations dict for classes
# explanations_dict_classes = {}
# for class_nr in range(len(classes)):
#     explanations_dict_classes[classes[class_nr]]\
#         = explanations.as_list(class_nr)

# sort by entity
explanations_dict_entities_class = {}
for cls in explanations_dict_class_entities.items():
    for ent in cls[1].items():
        try:
            explanations_dict_entities_class[ent[0]] is not None
        except:
            explanations_dict_entities_class[ent[0]] = {}
        explanations_dict_entities_class[ent[0]][cls[0]] = ent[1]

# sort class_entity and entity_class index
sorted_class_entity_index = {}
for cls in explanations_dict_class_entities.items():
    sorted_class_entity_index[cls[0]] = dict(sorted(cls[1].items(), key=lambda item: item[1],reverse=True))
sorted_entity_class_index = {}
for ent in explanations_dict_entities_class.items():
    sorted_entity_class_index[ent[0]] = dict(sorted(ent[1].items(), key=lambda item: item[1],reverse=True))

# save explanations dict for entities
with open(outputPath + "most_discriminative_" + mode + "_class_entity.json","w",
          encoding='utf8') as f:
    json.dump(sorted_class_entity_index,f)
# save explanations dict for classes
with open(outputPath + "most_discriminative_" + mode + "_entity_class.json","w",
          encoding='utf8') as f:
    json.dump(sorted_entity_class_index,f)

# # print prediction and explanation
# class_nr = 5
# print("True label: " + example_lab)
# print("Confidence: " + str(pipeline.predict_proba([example_text])))
# print("Explanation for class '" + docLabs[class_nr] + "': "
#       + str(explanations.as_list(label=class_nr)))
# #print([docLabs[index] for index in explanations.available_labels()])
# #explanations.show_in_notebook(text=False)

print("end")