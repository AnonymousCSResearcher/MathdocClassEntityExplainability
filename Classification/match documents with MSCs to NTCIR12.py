import pandas as pd
import os

# get file paths in arXiv dataset
dataset_path = r"D:\NTCIR-12_MathIR_arXiv_Corpus\NTCIR12"

folder_files = {}
for folder in os.listdir(dataset_path):
    folder_files[folder] = []
    for file in os.listdir(dataset_path + "/" + folder):
        folder_files[folder].append(file)

# load mscs table
filename = r"C:\Users\phili\Dropbox\PhD\Projects\Formula Clustering\FormulaFeatureAnalysis\docsecabsTextMathClassClust\Entities&Categories\MSCclassification\msc_categories.csv"
table = pd.read_csv(filename,sep=r'\s*,\s*',dtype=str)

for index in range(0,len(table)):
    arXiv_id = table['arxiv_id'].iloc[index]

    prefix,postfix = arXiv_id.split(".")
    try:
        candidates = folder_files[prefix]
        for candidate in candidates:
            if arXiv_id == candidate.strip(".tei"):
                print(arXiv_id + "=?" + candidate)
    except:
        pass

print("end")