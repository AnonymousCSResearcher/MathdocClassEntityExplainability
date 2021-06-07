import pickle
import statistics
import numpy as np
from scipy import stats

#global path
mypath = "F:\\NTCIR-12_MathIR_arXiv_Corpus\\output_AnnoTeX"

with open(mypath + "\\identifier_statistics.pkl", "rb") as f:
    identifier_statistics = pickle.load(f)

###

# get identifier_class_distribution_scores
identifier_class_distribution_scores = {}
for identifier in identifier_statistics['identifier_class_distribution'].items():
    identifier_class_distribution_scores[identifier[0]] = {}
    for classs in identifier[1].items():
        identifier_class_distribution_scores[identifier[0]][classs[0]]\
            = classs[1]/identifier_statistics['class_distribution'][classs[0]]

# get semantics_class_distribution_scores
semantics_class_distribution_scores = {}
for semantics in identifier_statistics['semantics_class_distribution'].items():
    semantics_class_distribution_scores[semantics[0]] = {}
    for classs in semantics[1].items():
        semantics_class_distribution_scores[semantics[0]][classs[0]]\
            = classs[1]/identifier_statistics['class_distribution'][classs[0]]

# get identifier variances
identifier_variances = {}
for identifier in identifier_class_distribution_scores.items():
    identifier_variances[identifier[0]]\
        = statistics.variance([classs[1] for classs in identifier[1].items()])
identifier_variances_mean = np.mean([identifier[1] for identifier in identifier_variances.items()])
identifier_variances_min = min([identifier[1] for identifier in identifier_variances.items()])
identifier_variances_max = max([identifier[1] for identifier in identifier_variances.items()])

# get semantics variances
semantics_variances = {}
for semantics in semantics_class_distribution_scores.items():
    try:
        semantics_variances[semantics[0]]\
            = statistics.variance([classs[1] for classs in semantics[1].items()])
    except:
        pass
semantics_variances_mean = np.mean([semantics[1] for semantics in semantics_variances.items()])
semantics_variances_min = min([semantics[1] for semantics in semantics_variances.items()])
semantics_variances_max = max([semantics[1] for semantics in semantics_variances.items()])

# get identifier entropies
identifier_entropies = {}
for identifier in identifier_class_distribution_scores.items():
    identifier_entropies[identifier[0]]\
        = stats.entropy([classs[1] for classs in identifier[1].items()],base=2)
identifier_entropies_mean = np.mean([identifier[1] for identifier in identifier_entropies.items()])
identifier_entropies_min = min([identifier[1] for identifier in identifier_entropies.items()])
identifier_entropies_max = max([identifier[1] for identifier in identifier_entropies.items()])

# get semantics entropies
semantics_entropies = {}
for semantics in semantics_class_distribution_scores.items():
    semantics_entropies[semantics[0]]\
        = stats.entropy([classs[1] for classs in semantics[1].items()],base=2)
semantics_entropies_mean = np.mean([semantics[1] for semantics in semantics_entropies.items()])
semantics_entropies_min = min([semantics[1] for semantics in semantics_entropies.items()])
semantics_entropies_max = max([semantics[1] for semantics in semantics_entropies.items()])

print("end")