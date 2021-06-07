import pickle
from sklearn.metrics import accuracy_score, confusion_matrix

path = "E:\\NTCIR-12_MathIR_arXiv_Corpus\\ML_output_balanced\\arXivEmbeddings\\"
labs_file = path + "secLabs_sixclass.pkl"
wiki_labs_file = path + "WikiLabs_sixclass.pkl"
labs_entropies_file = path + "WikiLabsEntropies.pkl"

with open(labs_file,"rb") as f:
    labs = pickle.load(f)

with open(wiki_labs_file,"rb") as f:
    wiki_labs = pickle.load(f)

with open(labs_entropies_file,"rb") as f:
    labs_entropies = pickle.load(f)

# normalize
#labs_entropies = list(labs_entropies/max(labs_entropies))

# count matches
matches = 0
i = 0
for lab in labs:
    j = 0
    for wiki_lab in wiki_labs:
        if wiki_lab == lab and i==j:
            matches += 1
        j += 1
    i += 1

accuracy = accuracy_score(labs,wiki_labs)
#conf_mat = confusion_matrix(labs,wiki_labs)

print(accuracy)