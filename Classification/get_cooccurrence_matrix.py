import numpy as np
import pandas as pd

# load table
filename = "msc_categories.csv"
table = pd.read_csv(filename,sep=r'\s*,\s*')

# create cooccurrence catalog
cooccurrences = {}
msc_collector = set()
category_collector = set()
for index in range(0,len(table)):
    mscs = table['msc'].iloc[index]
    categories = table['categories'].iloc[index]
    for msc in mscs.split():
        msc_collector.add(msc)
        for category in categories.split():
            category_collector.add(category)
            try:
                cooccurrences[msc][category] += 1
            except:
                try:
                    cooccurrences[msc][category] = 1
                except:
                    cooccurrences[msc] = {}
                    cooccurrences[msc][category] = 1
            try:
                cooccurrences[category][msc] += 1
            except:
                try:
                    cooccurrences[category][msc] = 1
                except:
                    cooccurrences[category] = {}
                    cooccurrences[category][msc] = 1

# create cooccurrence matrix
cooccurrence_matrix = np.zeros(shape=(len(msc_collector),len(category_collector)))
row_idx = 0
for msc in msc_collector:
   col_idx = 0
   for category in category_collector:
       try:
           cooccurrence_matrix[col_idx][row_idx] = cooccurrences[msc][category]
       except:
           pass
       col_idx += 1
   row_idx += 1

cooccurrence_matrix = pd.DataFrame(data=cooccurrence_matrix,index=msc_collector,columns=category_collector,dtype=int)

# write to csv
cooccurrence_matrix.to_csv('cooccurrence_matrix.csv')

print("end")