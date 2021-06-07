import pandas as pd
import numpy as np

# load table
filename = "msc_categories.csv"
table = pd.read_csv(filename,sep=r'\s*,\s*')
data = table[['msc','categories']]
del table

# # get unique mscs
# mscs = set()
# for index in range(0,len(data)):
#     mscs_text = data['msc'].iloc[index]
#     for msc in mscs_text.split():
#         mscs.add(msc)

# get unique arXiv categories
categories = set()
for index in range(0,len(data)):
    categories_text = data['categories'].iloc[index]
    for category in categories_text.split():
        categories.add(category.split(".")[0])

# prepare OneVsAll classification
# set each category as binary labeled column
# init with zeros
for category in categories:
    data[category] = 0
# fill ones
for index in range(0,len(data)):
    categories_text = data['categories'].iloc[index]
    for category in categories_text.split():
        data[category].iloc[index] = 1

# classify
# test with first class

print("end")