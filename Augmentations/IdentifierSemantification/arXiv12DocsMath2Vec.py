from os import listdir
import pickle

from bs4 import BeautifulSoup

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

                        # retrieve math data (formulae) from document
                        with open(datasetPath + "\\" + Dir + "\\" + File, "r", encoding="utf8") as f:
                            formulae = BeautifulSoup(f.read(), 'html.parser').find_all('formula')

                        # store filename
                        docNames.append(File)
                        # store label
                        docLabs.append(classLab)

                        # extract operators/identifiers from formulae
                        # Data = set() # operator/identifier set for doc2vec
                        Data = []  # operator/identifier list for doc2vec
                        for formula in formulae:

                            formulaString = str(formula.contents)


                            def findall(p, s):
                                '''Yields all the positions of
                                the pattern p in the string s.'''
                                i = s.find(p)
                                while i != -1:
                                    yield i
                                    i = s.find(p, i + 1)


                            # retrieve operators
                            for i in findall('</m:mo', formulaString):
                                try:
                                    tmp = formulaString[i - 5:i]
                                    # character can be formula operator or identifier
                                    character = tmp.split('>')[1]
                                    # print(character)
                                    Data.append(character)
                                except:
                                    pass
                            # retrieve identifiers
                            for i in findall('</m:mi', formulaString):
                                try:
                                    tmp = formulaString[i - 5:i]
                                    # character can be formula operator or identifier
                                    character = tmp.split('>')[1]
                                    # print(character)
                                    Data.append(character)
                                except:
                                    pass
                        # docData.append(list(Data))
                        docData.append(Data)

# Save document data

# try:
#     with open(outputPath + "secDataMath.pkl",'wb') as f:
#         pickle.dump(docData, f)
# except:
#     print("Failed to save data!")

# Build Doc2Vec math model

# model,doc2vecMath = docs2vec(docData,docNames)

# Save Doc2Vec math model

# try:
#     with open(outputPath + "doc2vecMath_op.model", 'wb') as f:
#         pickle.dump(model, f)
# except:
#     print("Failed to save model!")

# Save document labels and math vectors

try:
    with open(outputPath + "secLabsMath_sixclass.pkl",'wb') as f:
        pickle.dump(docLabs, f)
except:
    print("Failed to save labels!")

# try:
#     with open(outputPath + "sec2vecMath_id.pkl",'wb') as f:
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
    with open(outputPath + "secMath_sixclass_opid_tfidf.pkl",'wb') as f:
        pickle.dump(docMath_tfidf, f)
except:
    print("Failed to save math vectors!")

print("end")