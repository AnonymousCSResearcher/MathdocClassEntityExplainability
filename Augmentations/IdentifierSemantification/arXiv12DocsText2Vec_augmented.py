from os import listdir
import pickle

from nltk import RegexpTokenizer
from nltk.corpus import stopwords

from Augmentations.Wikipedia_Categories_Concepts import get_category_from_concepts, \
    get_category_concepts_augmentations

#from arXivDocs2Vec import docs2vec
from arXivDocs2tfidf import docs2tfidf

# Set file paths
basePath = "E:\\NTCIR-12_MathIR_arXiv_Corpus\\"

datasetPath = basePath + "NTCIR12\\"
#valid_folder_prefix = ["0001"]
#valid_folder_prefix = ["00", "01", "02", "03", "04", "05", "06"]
#valid_folder_prefix = ["00", "01"]
valid_folder_prefix = [""]

outputPath = basePath + "ML_output_balanced\\arXivEmbeddings\\"

# Create lists for document data and labels

docData = []
docNames = []
docLabs = []

# Define class limit and desired classes
classCounter = {} #14*250 = 3500
# arXiv 2012
#classLimit = 250
classLimit = 100
#desired_classes = ['astro-ph', 'cond-mat', 'cs', 'gr-qc', 'hep-lat', 'hep-ph', 'hep-th', 'math-ph', 'math', 'nlin', 'quant-ph', 'physics', 'alg-geom', 'q-alg']
desired_classes = ['astro-ph', 'cond-mat', 'gr-qc', 'hep-ph', 'quant-ph', 'physics']
#desired_classes = ['cs', 'math', 'physics']

# Define data cleaning

tokenizer = RegexpTokenizer(r'\w+')
stopword_set = set(stopwords.words('english'))

#store Wikipedia KG labels and entropies
WikiLabs = []
LabsEntropies = []

def nlp_clean(raw_str):

    lowered_str = raw_str.lower()
    tokenized_str = tokenizer.tokenize(lowered_str)
    swremoved = list(set(tokenized_str).difference(stopword_set))
    cleaned = [] # word list for doc2vec
    for word in swremoved:
        contains_digit = False
        for char in word:
            if char.isdigit():
                contains_digit = True
        if not contains_digit and len(word) > 3:
            cleaned.append(word) # word list for doc2vec
    #print(cleaned)
    #get Wikipedia KG label
    category,entropy = get_category_from_concepts(cleaned)
    WikiLabs.append(category)
    LabsEntropies.append(entropy)
    #augment with Wikipedia Concepts Category matching
    return get_category_concepts_augmentations(cleaned)
    #cleaned.extend(get_category_concepts_augmentations(cleaned))
    #return cleaned

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

                        # retrieve cleaned text data from document
                        # cleaned
                        docData.append(nlp_clean(open(datasetPath + "\\" + Dir + "\\" + File, "r", encoding="utf8").read()))
                        # uncleaned
                        #docData.append(open(datasetPath + "\\" + Dir + "\\" + File, "r", encoding="utf8").read().split())

                        # store filename
                        docNames.append(File)
                        # store label
                        docLabs.append(classLab)

# Save document data

# try:
#     with open(outputPath + "secDataText.pkl",'wb') as f:
#         pickle.dump(docData, f)
# except:
#     print("Failed to save data!")

# Build Doc2Vec text model

# model,doc2vecText = docs2vec(docData,docNames)

# Save Doc2Vec text model

# try:
#     with open(outputPath + "doc2vecText.model", 'wb') as f:
#         pickle.dump(model, f)
# except:
#     print("Failed to save model!")

# Save document names, labels and text vectors

# try:
#     with open(outputPath + "secNamesText.pkl",'wb') as f:
#         pickle.dump(docNames, f)
# except:
#     print("Failed to save names!")

try:
    with open(outputPath + "secLabs_sixclass.pkl",'wb') as f:
        pickle.dump(docLabs, f)
    with open(outputPath + "WikiLabs_sixclass.pkl",'wb') as f:
        pickle.dump(WikiLabs, f)
    with open(outputPath + "WikiLabsEntropies.pkl",'wb') as f:
        pickle.dump(LabsEntropies, f)
except:
    print("Failed to save labels!")

# try:
#     with open(outputPath + "sec2vecText_uncleaned.pkl",'wb') as f:
#         pickle.dump(doc2vecText, f)
# except:
#     print("Failed to save text vectors!")

# Build and save tf_idf text vectors

# document strings for tfidf
docData_strings = []
for doc in docData:
    docString = ""
    for word in doc:
        docString += word + " "
    # remove whitespace at the end
    docData_strings.append(docString[:-1])

docText_tfidf = docs2tfidf(docData_strings)

try:
    with open(outputPath + "secText_sixclass_categoryconcept_tfidf.pkl",'wb') as f:
        pickle.dump(docText_tfidf, f)
except:
    print("Failed to save text vectors!")

print("end")