import pandas as pd
import numpy as np
import json

arXiv_file = "uncertainties_arXiv.json"
MSCs_file = "uncertainties_MSCs.json"

# load table and uncertainties dict
table = pd.read_csv("cooccurrence_matrix.csv")
with open(arXiv_file,"r") as f:
    uncertainties = json.load(f)

# compute mean, min, and max (entropies)
entropies = []
for entropy in uncertainties['entropies'].items():
    if str(entropy[1]) != 'nan':
        entropies.append(entropy[1])
print("Mean entropy: " + str(np.mean(entropies)))
print("Min entropy: " + str(min(entropies)))
print("Max entropy: " + str(max(entropies)))

# compute mean, min, and max (margins)
margins = []
for margin in uncertainties['margins'].items():
    margins.append(margin[1])
print("Mean margin: " + str(np.mean(margins)))
print("Min margin: " + str(min(margins)))
print("Max margin: " + str(max(margins)))

print("end")