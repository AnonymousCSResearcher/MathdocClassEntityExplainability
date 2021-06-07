import json
from scipy import stats
import numpy as np

# Set file paths
basePath = 'D:\\NTCIR-12_MathIR_arXiv_Corpus\\'
inputPath = basePath + 'output_Explainability\\100perClass\\'
outputPath = inputPath

# Load dicts
with open(outputPath + "most_discriminative_text_class_entity.json","r",) as f:
    mdisc_text_cls_ent = json.load(f)
with open(outputPath + "most_discriminative_text_entity_class.json","r",) as f:
    mdisc_text_ent_cls = json.load(f)
with open(outputPath + "most_frequent_text_class_entity.json","r") as f:
    mfreq_text_cls_ent = json.load(f)
with open(outputPath + "most_frequent_text_entity_class.json","r") as f:
    mfreq_text_ent_cls = json.load(f)
with open(outputPath + "most_discriminative_math_class_entity.json","r",) as f:
    mdisc_math_cls_ent = json.load(f)
with open(outputPath + "most_discriminative_math_entity_class.json","r",) as f:
    mdisc_math_ent_cls = json.load(f)
with open(outputPath + "most_frequent_math_class_entity.json","r") as f:
    mfreq_math_cls_ent = json.load(f)
with open(outputPath + "most_frequent_math_entity_class.json","r") as f:
    mfreq_math_ent_cls = json.load(f)

# Calculate entropies
def get_entropies(stat_dict):
    name = stat_dict[0]
    dict = stat_dict[1]
    entropies = []
    for key in dict.items():
        entropy = stats.entropy([entry[1] for entry in key[1].items()])
        #print(key[0] + ": " + str(entropy))
        entropies.append(entropy)
    print(name + ": " + str(np.mean(entropies)))

# Name and list stat dicts
stat_dicts = [("mdisc_text_cls_ent",mdisc_text_cls_ent),
              ("mdisc_text_ent_cls", mdisc_text_ent_cls),
              ("mfreq_text_cls_ent", mfreq_text_cls_ent),
              ("mfreq_text_ent_cls", mfreq_text_ent_cls),
              ("mdisc_math_cls_ent", mdisc_math_cls_ent),
              ("mdisc_math_ent_cls", mdisc_math_ent_cls),
              ("mfreq_math_cls_ent", mfreq_math_cls_ent),
              ("mfreq_math_ent_cls", mfreq_math_ent_cls)]

# Compare entropies of dict modes
for stat_dict in stat_dicts:
    get_entropies(stat_dict)

# # Compare entropies among classes or entities
# label = "acceleration"
# for stat_dict in stat_dicts:
#     name = stat_dict[0]
#     dict = stat_dict[1]
#     try:
#         entropy = stats.entropy([entry[1] for entry in dict[label].items()])
#         print(name + "[" + label + "]: " + str(entropy))
#     except:
#         pass

# Compare entropies among sampling (stat_dict) over classes
#stat_dict = mdisc_text_cls_ent
#for cls in stat_dict.items():
#    entropy = stats.entropy([entry[1] for entry in cls[1].items()])
#    print(cls[0] + ": " + str(entropy))

print("end")