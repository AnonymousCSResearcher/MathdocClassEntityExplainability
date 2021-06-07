import pandas as pd
import json

# load cooccurrence matrix
filename = "cooccurrence_matrix.csv"
cooccurrence_matrix = pd.read_csv(filename,sep=r'\s*,\s*')

# predictions
predictions = {}
index_column = 'Unnamed: 0'
n_MSCdigits = 2
for column in list(cooccurrence_matrix.columns):
    if column != index_column:
        prediction_index = cooccurrence_matrix[column].argmax()
        predictions[column] = cooccurrence_matrix.iloc[prediction_index,
                                                       cooccurrence_matrix.columns.get_loc(index_column)][:n_MSCdigits]

# save to json
with open("predictions_coocurrence.json",'w') as f:
    json.dump(predictions,f)

print("end")