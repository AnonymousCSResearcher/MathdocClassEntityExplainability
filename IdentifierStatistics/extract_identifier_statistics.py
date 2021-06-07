import pickle
from operator import itemgetter
from collections import OrderedDict

#global path
mypath = "F:\\NTCIR-12_MathIR_arXiv_Corpus\\output_AnnoTeX"

# annotation_catalog = {}
with open(mypath + "\\annotation_catalog_all.pkl", "rb") as f:
    annotation_catalog = pickle.load(f)

# find surrounding words as annotation candidates

# exclude formulae and stopwords from candidates
excluded = [">", "<", "=", '"']
with open(mypath + "\\stopwords.txt") as f:
    stopwords = [line.strip() for line in f]
# only Latin and Greek letters
with open(mypath + "\\valid.txt") as f:
    valid = [line.strip() for line in f]

# list annotation candidate occurences for each identifier
identifier_statistics = {}

class_distribution = {}

identifier_class_distribution = {}
class_identifier_distribution = {}

class_semantics_distribution = {}
semantics_class_distribution = {}

identifier_class_semantics_distribution = {}
semantics_identifier_distribution = {}
semantics_class_identifier_distribution = {}

for identifier in annotation_catalog.items():
    # exclude identifiers with indices
    #if not "_" in identifier[0] and not "^" in identifier[0]:
    # include only Latin and Greek letters
    if identifier[0] in valid:
        matches = {}
        # split candidate text sentence into words
        for candidate in identifier[1].items():
            classs = candidate[0][:-11]

            # update class distribution
            try:
                class_distribution[classs] += 1
            except:
                class_distribution[classs] = 1

            # update identifier class distribution
            try:
                identifier_class_distribution[identifier[0]][classs] += 1
            except:
                try:
                    identifier_class_distribution[identifier[0]][classs] = 1
                except:
                    identifier_class_distribution[identifier[0]] = {}
                    identifier_class_distribution[identifier[0]][classs] = 1

            # update class identifier distribution
            try:
                class_identifier_distribution[classs][identifier[0]] += 1
            except:
                try:
                    class_identifier_distribution[classs][identifier[0]] = 1
                except:
                    class_identifier_distribution[classs] = {}
                    class_identifier_distribution[classs][identifier[0]] = 1

            # update semantics class distribution
            # classs instead of class because of python ;)
            words = candidate[1].split()
            for word in words:
                # lowercase and remove .,-()
                word = word.lower()
                char_excl = [".", ",", "-", "(", ")"]
                for c in char_excl:
                    word = word.replace(c, "")
                # not part of a formula environment
                not_formula = not True in [ex in word for ex in excluded]
                # not stopword
                not_stopword = word not in stopwords
                if not_formula and not_stopword:
                    # count occurences
                    try:
                        matches[classs][word] += 1
                    except:
                        try:
                            matches[classs][word] = 1
                        except:
                            matches[classs] = {}
                            matches[classs][word] = 1

                    # update class semantics distribution
                    try:
                        class_semantics_distribution[classs][word] += 1
                    except:
                        try:
                            class_semantics_distribution[classs][word] = 1
                        except:
                            class_semantics_distribution[classs] = {}
                            class_semantics_distribution[classs][word] = 1
                    
                    # update semantics class distribution
                    try:
                        semantics_class_distribution[word][classs] += 1
                    except:
                        try:
                            semantics_class_distribution[word][classs] = 1
                        except:
                            semantics_class_distribution[word] = {}
                            semantics_class_distribution[word][classs] = 1

                    # update semantics identifier distribution
                    try:
                        semantics_identifier_distribution[word][identifier[0]] += 1
                    except:
                        try:
                            semantics_identifier_distribution[word][identifier[0]] = 1
                        except:
                            semantics_identifier_distribution[word] = {}
                            semantics_identifier_distribution[word][identifier[0]] = 1

                    # update semantics class identifier distribution
                    try:
                        semantics_class_identifier_distribution[word][classs][identifier[0]] += 1
                    except:
                        try:
                            semantics_class_identifier_distribution[word][classs][identifier[0]] = 1
                        except:
                            try:
                                semantics_class_identifier_distribution[word][classs] = {}
                                semantics_class_identifier_distribution[word][classs][identifier[0]] = 1
                            except:
                                semantics_class_identifier_distribution[word] = {}
                                semantics_class_identifier_distribution[word][classs] = {}
                                semantics_class_identifier_distribution[word][classs][identifier[0]] = 1

        # sort matches for each classs
        sorted_matches = {}
        for classss in matches.items():
            sorted_matches[classss[0]] = OrderedDict(
                sorted(classss[1].items(), key=itemgetter(1), reverse=True))

        identifier_class_semantics_distribution[identifier[0]] = sorted_matches

        # update identifier statistics
        identifier_statistics['class_distribution'] = class_distribution
        identifier_statistics['identifier_class_distribution'] = identifier_class_distribution
        identifier_statistics['class_identifier_distribution'] = class_identifier_distribution
        identifier_statistics['identifier_class_semantics_distribution'] = identifier_class_semantics_distribution
        identifier_statistics['class_semantics_distribution'] = class_semantics_distribution
        identifier_statistics['semantics_class_distribution'] = semantics_class_distribution
        identifier_statistics['semantics_identifier_distribution'] = semantics_identifier_distribution
        identifier_statistics['semantics_class_identifier_distribution'] = semantics_class_identifier_distribution

        with open(mypath + "\\identifier_statistics.pkl", "wb") as f:
            pickle.dump(OrderedDict(sorted(identifier_statistics.items())), f)