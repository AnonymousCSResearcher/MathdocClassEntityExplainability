import pandas as pd
from scipy import stats
from collections import Counter
import json

# load cooccurrence matrix
filename = "cooccurrence_matrix.csv"
cooccurrence_matrix = pd.read_csv(filename,sep=r'\s*,\s*')

def get_uncertainties_columns():
    # uncertainties
    uncertainties = {}
    uncertainties['classes'] = {}
    counters = {}
    entropies = {}
    margins = {}
    for column in list(cooccurrence_matrix.columns):
        if column != 'Unnamed: 0':
            uncertainties['classes'][column] = {}
            # counter
            counter = Counter(cooccurrence_matrix[column])
            uncertainties['classes'][column]['counter'] = counter
            counters[column] = counter
            # (shannon) entropy
            entropy = stats.entropy(cooccurrence_matrix[column])
            uncertainties['classes'][column]['entropy'] = entropy
            entropies[column] = entropy
            # margin (between first and second most frequent)
            # get first and second max
            sort = sorted(counter.items())
            first_max, second_max = sort[0][1], sort[1][1]
            # difference
            margin = (first_max - second_max) / sum([count[1] for count in counter.items()])
            # normalized difference
            #margin = (first_max-second_max)/first_max # = (1-second_max/first_max)
            uncertainties['classes'][column]['margin'] = margin
            margins[column] = margin

    # add entropies and margins to uncertainties dict
    uncertainties['entropies'] = entropies
    uncertainties['margins'] = margins

    return uncertainties

def get_uncertainties_rows():
    # uncertainties
    uncertainties = {}
    uncertainties['classes'] = {}
    counters = {}
    entropies = {}
    margins = {}
    for idx,cont in cooccurrence_matrix.iterrows():
        if idx != 0:
            row = cont[0]
            uncertainties['classes'][row] = {}
            # counter
            counter = Counter(cont[1:-1])
            uncertainties['classes'][row]['counter'] = counter
            counters[row] = counter
            # (shannon) entropy
            entropy = stats.entropy(list(cont[1:-1]))
            uncertainties['classes'][row]['entropy'] = entropy
            entropies[row] = entropy
            # margin (between first and second most frequent)
            # get first and second max
            sort = sorted(counter.items())
            try:
                first_max, second_max = sort[0][1], sort[1][1]
                margin = (first_max - second_max) / sum([count[1] for count in counter.items()])
            except:
                margin = 0
            # difference

            # normalized difference
            # margin = (first_max-second_max)/first_max # = (1-second_max/first_max)
            uncertainties['classes'][row]['margin'] = margin
            margins[row] = margin

    # add entropies and margins to uncertainties dict
    uncertainties['entropies'] = entropies
    uncertainties['margins'] = margins

    return uncertainties

# get uncertainties for arXiv and MSCs
uncertainties_arXiv = get_uncertainties_columns()
uncertainties_MSCs = get_uncertainties_rows()

# save dict to json
with open("uncertainties_arXiv.json","w") as f:
    json.dump(uncertainties_arXiv,f)
with open("uncertainties_MSCs.json","w") as f:
    json.dump(uncertainties_MSCs,f)

print("end")