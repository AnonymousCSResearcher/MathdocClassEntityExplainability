from Wikipedia_Categories_Concepts_Data import category_concepts_dict
from collections import Counter
import operator
from scipy import stats

def get_category_concepts_augmentations(text):
    augmentations = []
    for category in category_concepts_dict.items():
        for word in text:
            if word in category[1]:
                augmentations.append(category[0])
    class_counter = Counter(augmentations)
    major_class = max(class_counter.items(), key=operator.itemgetter(1))[0]
    #text.extend([major_class for i in range(0,class_counter[major_class])])
    #text.extend(major_class)
    return major_class

def get_category_from_concepts(text):
    augmentations = []
    for category in category_concepts_dict.items():
        for word in text:
            if word in category[1]:
                augmentations.append(category[0])
    try:
        class_counter = Counter(augmentations)
        major_class = max(class_counter.items(), key=operator.itemgetter(1))[0]
        entropy = stats.entropy(
            [category[1] for category in class_counter.items()], base=2)
    except:
        major_class = 'None'
        entropy = 'N/A'
    return major_class,entropy