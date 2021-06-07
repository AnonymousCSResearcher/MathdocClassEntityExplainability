import pandas as pd
from scipy import stats
from collections import Counter

# load cooccurrence matrix
filename = "cooccurrence_matrix.csv"
cooccurrence_matrix = pd.read_csv(filename,sep=r'\s*,\s*')

print(Counter(cooccurrence_matrix['math.SG']))
print(stats.entropy(cooccurrence_matrix['math.SG']))

print("end")