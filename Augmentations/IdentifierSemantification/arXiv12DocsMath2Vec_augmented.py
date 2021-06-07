from os import listdir
import pickle
import json

from nltk import RegexpTokenizer
from nltk.corpus import stopwords

from bs4 import BeautifulSoup
import re

#from arXivDocs2Vec import docs2vec
from arXivDocs2tfidf import docs2tfidf

# Load MathWikiLink/AnnoMathTeX identifier semantification sources
sources_path = "E:\\MathWikiLink\\sources\\"
# Open sources/identifier_name_recommendations
sources = ['Wikidata', 'Wikipedia', 'arXiv']
source_dicts = {}
for source in sources:
    with open(sources_path + '/' + 'identifier_name_recommendations_' + source + '.json', 'r') as f:
        source_dicts[source] = json.load(f)

# Set file paths

datasetPath = "E:\\NTCIR-12_MathIR_arXiv_Corpus\\NTCIR12\\"
#valid_folder_prefix = ["0001"]
#valid_folder_prefix = ["00", "01", "02", "03", "04", "05", "06"]
#valid_folder_prefix = ["00", "01"]
valid_folder_prefix = [""]

outputPath = "E:\\NTCIR-12_MathIR_arXiv_Corpus\\ML_output_balanced\\arXivEmbeddings\\"

# Create lists for data, (math) labels and math vectors

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

# exclude formulae, stopwords, html and letters from candidates
excluded = [">", "<", "=", "~",'"', "_"]
with open("stopwords.txt") as f:
    stopwords_math = [line.strip() for line in f]
#invalid = ["times"]
with open("letters.txt") as f:
    letters = [line.strip() for line in f]

# Define data cleaning

tokenizer = RegexpTokenizer(r'\w+')
stopword_set_text = set(stopwords.words('english'))

def nlp_clean(raw_str):

    lowered_str = raw_str.lower()
    tokenized_str = tokenizer.tokenize(lowered_str)
    swremoved = list(set(tokenized_str).difference(stopword_set_text))
    cleaned = [] # word list for doc2vec
    for word in swremoved:
        contains_digit = False
        for char in word:
            if char.isdigit():
                contains_digit = True
        if not contains_digit and len(word) > 3:
            cleaned.append(word) # word list for doc2vec
    #print(cleaned)
    return cleaned

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

                        Data = []  # list of words

                        # retrieve cleaned text data from document
                        # cleaned
                        #Data.extend(
                        #    nlp_clean(open(datasetPath + "\\" + Dir + "\\" + File, "r", encoding="utf8").read()))
                        # uncleaned
                        # docData.append(open(datasetPath + "\\" + Dir + "\\" + File, "r", encoding="utf8").read().split())

                        # retrieve math data (formulae) from document
                        with open(datasetPath + "\\" + Dir + "\\" + File, "r", encoding="utf8") as f:
                            filestring = f.read()
                            formulae = BeautifulSoup(filestring, 'html.parser').find_all('formula')

                        # store filename
                        docNames.append(File)
                        # store label
                        docLabs.append(classLab)


                        #extract identifiers from formulae
                        for formula in formulae:

                            formulaString = str(formula.contents)

                            def findall(p, s):
                                '''Yields all the positions of
                                the pattern p in the string s.'''
                                i = s.find(p)
                                while i != -1:
                                    yield i
                                    i = s.find(p, i + 1)

                            # retrieve formula

                            # extract TeX formula
                            # formulaString
                            s = str(formula.contents)
                            start = 'alttext="'
                            end = '" display='
                            try:
                                #TeX = re.search('%s(.*)%s' % (start, end), s).group(1)
                                TeX = formula.contents[0].attrs['alttext']
                            except:
                                TeX = ""

                            # retrieve identifiers
                            ids = []
                            for i in findall('</m:mi', str(formula.contents)):
                                try:
                                    tmp = formulaString[i - 5:i]
                                    # character can be formula operator or identifier
                                    character = tmp.split('>')[1]
                                    # print(character)
                                    ids.append(character)
                                except:
                                    pass

                            # semantify identifiers using MathWikiLink/AnnoMathTeX sources
                            for id in ids:
                                source = 'arXiv'
                                try:
                                    Data.extend(source_dicts[source][id])
                                except:
                                    pass
                                source = 'Wikipedia'
                                try:
                                    Data.extend([item['description'] for item in source_dicts[source][id]])
                                except:
                                    pass
                                source = 'Wikidata'
                                try:
                                    Data.extend([item['name'] for item in source_dicts[source][id]])
                                except:
                                    pass

                            # extract surrounding tex
                            index = filestring.find('alttext="' + TeX + '" display=')
                            surrounding_text_candidates = filestring[index - 500:index + 500]

                            for word in surrounding_text_candidates.split():
                                # lowercase and remove .,-()
                                word = word.lower()
                                char_excl = [".", ":", ",", "-", "(", ")"]
                                for c in char_excl:
                                    word = word.replace(c, "")
                                # not part of a formula environment
                                not_formula = not True in [ex in word for ex in excluded]
                                # not stopword
                                not_stopword = word not in stopwords_math
                                # not invalid html
                                #not_invalid = not True in [inv in word for inv in invalid]
                                # not a latin or greek letter
                                not_letter = word not in letters
                                if not_formula and not_stopword and not_letter: #and not_invalid
                                    Data.append(word)

                        docData.append(Data)

# Build Doc2Vec math model

#model,doc2vecMath = docs2vec(docData,docNames)

# Save Doc2Vec math model

# try:
#     with open(outputPath + "sec2vecMath_semantics_uncleaned.model", 'wb') as f:
#         pickle.dump(model, f)
# except:
#     print("Failed to save model!")

# Save document labels and math vectors

try:
    with open(outputPath + "secLabsMath_sixclass2.pkl",'wb') as f:
        pickle.dump(docLabs, f)
except:
    print("Failed to save labels!")

# try:
#     with open(outputPath + "sec2vecMath_semantics.pkl",'wb') as f:
#         pickle.dump(doc2vecMath, f)
# except:
#     print("Failed to save math vectors!")

# Build and save tf_idf math vectors

# document strings for tfidf
docData_strings = []
for doc in docData:
    docString = ""
    for word in doc:
        docString += word + " "
    # remove whitespace at the end
    docData_strings.append(docString[:-1])

docMath_tfidf = docs2tfidf(docData_strings)

try:
    with open(outputPath + "secText_sixclass_annomathtex_tfidf2.pkl",'wb') as f:
        pickle.dump(docMath_tfidf, f)
except:
    print("Failed to save math vectors!")

print("end")