from paths import formula_catalog_path
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

with open(formula_catalog_path,'rb') as f:
    formula_catalog = pickle.load(f)

del formula_catalog['operator_catalog']
del formula_catalog['identifier_catalog']

def count(dict,key):
    try:
        dict[key] += 1
    except:
        dict[key] = 1
    return dict

# LIST FORMULA DUPLICATE COUNTS
formula_duplicates_dict = {}
for formula in formula_catalog.items():
    try:
        key = formula[1]['TeX']
        #if len(key.split()) > 1:
        if "=" in key and len(key.split()) > 2:
            count(formula_duplicates_dict,key)
    except:
        pass

formula_duplicates_dict = sorted(formula_duplicates_dict.items(), key=lambda x: x[1], reverse=True)


def docs2tfidf(docData):

    # Generate tf_idf vectors

    vectorizer = TfidfVectorizer()
    docVecs = vectorizer.fit_transform(docData)

    return docVecs

# CREATE FORMULA FEATURE VECTORS (TF-IDF)

formula_strings = []
formula_labs = []
for formula in formula_catalog.items():
    formula_string = ""
    for operator in formula[1]['operators'].items():
        formula_string += operator[1] + " "
    for identifier in formula[1]['identifiers'].items():
        formula_string += identifier[1] + " "
    formula_strings.append(formula_string[:-1])
    formula_labs.append(formula[1]['filename'].split("0")[0])

formula_vecs = docs2tfidf(formula_strings)

print("end")