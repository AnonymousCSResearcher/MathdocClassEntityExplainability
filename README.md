# Mathematical Entity Linking for STEM Document Classification Explainability

This manual provides descriptions to reproduce the results for each subsection of the evaluation section of the associated paper.

## Evaluation Workflow

![EvaluationWorkflow](https://github.com/AnonymousCSResearcher/STEMdocClassExplainability/blob/main/Workflow_EL-ClassExplainability.png)

## MSC-arXiv Category Correspondence

Data and algorithms can be found in the folder 'Classification'.

1) The data for the prediction of the MSC-arXiv category correspondence is the table in the file
```
msc_categories.csv
```

2) To get the co-ocurrence matrix with 6202 MSC rows and 156 arXiv category columns, run
```
get_cooccurrence_matrix.py
```
yielding 'coocurrence_matrix.csc'.

3) To get the uncertainty/entropy analysis results run
```
get_classification_uncertainties.py
```
or
```
get_classification_entropies.py
```
yielding 'uncertainties_arXiv.json' and 'uncertainties_MSCs.json'.

4) The multi-label prediction of arXiv categories from MSCs can be made with
```
get_multilabel_classification.py
```

5) The prediction of the categories using a LogReg classifier on TFIDF encodings or the co-occurrence matrix can be made with
```
predict_categories_classifier.py
```
or
```
predict_categories_cooccurrence.py
```
respectively.

6) A comparison of the prediction results from 5) can be made running
```
compare_predictions_classifier_to_cooccurrence.py
```

## Identifier Class Semantics Distributions

Data and algorithms can be found in the folder 'IdentifierStatistics'.

0) The NTCIR-11/12 arXiv dataset can be obtained from
```
http://ntcir-math.nii.ac.jp/data
```

1) The identifier annotation index 'annotation_catalog_all.pkl' is created from the surrounding text of mathematical formula symbols in parallel multiprocesses running
```
extract_surroundingtext_multiprocess.py
```

2) The identifier statistics are extracted from the 'annotation_catalog_all.pkl' using
```
compare_predictions_classifier_to_cooccurrence.py
```

3) Visualization of the identifier symbol and name distributions can be made via
```
visualize_identifier_statistics.py
```

## Unsupervised Semantic Identifier Enrichment for Document Classification

Data and algorithms can be found in the folder 'Augmentations/IdentifierSemantification'.

0) For the creation of the formula identifier semantification catalog see the 'Identifier Class Semantics Distributions' section with the script
```
extract_surroundingtext_multiprocess.py
```

1) The formula identifier catalog index can be analyzed for duplicates and encoded in TFID feature vectors using
```
analyze_formula_catalog.py
```

2) The encoding of the arXiv NTCIR-11/12 selection of 100 documents from each of the subject classes 'astro-ph', 'cond-mat', 'gr-qc', 'hep-lat', 'hep-ph', 'hep-th', 'math-ph', 'nlin', 'quant-ph', 'physics' is made via
```
arXiv12DocsText2Vec_augmented.py
```
for textual elements, and
```
arXiv12DocsMath2Vec_augmented.py
```
for mathematical elements (surrounding text of formulae).

3) The document classification is made and evaluated running
```
arXivClassification_semantified.py
```

## Category-Concept Augmentations

Data and algorithms can be found in the folder 'Augmentations'.

1) The data retrieved from https://en.wikipedia.org/w/index.php?title=Outline_of_physics&oldid=994270982 is condensed in
```
Wikipedia_Categories_Concepts_Data.py
```

2) The category concept augmentations for the text classification is carried out using
```
Wikipedia_Categories_Concepts.py
```

## Wikisource Entity Linking (Wikification)

Data and algorithms can be found in the folder 'EntityLinking'.

0) The processed documents are in the folder
```
documents/with_msc
```

1) To get the class-entity index run
```
get_class_entity_index(_Wikipedia).py
```

2) The Wikipedia article name dump can be accessed via
```
WikiDump.py
```

3) The evaluation of the entity linking (comparing eval modes 1-6) can be reproduced via
```
get_evaluation_entity_linking.py
```
with the binary scoring (TP, FP, FN, TN) made via
```
get_scoring_evaluation_entity_linking.py
```

4) An entropy index for 'text' or 'math' mode can be obtained via
```
get_index_entropies.py
```

## Class-Entity Explainability

Data and algorithms can be found in the folder 'Explainability'.

1) The most discriminative word features form the NTCIR-11/12 arXiv corpus (same selection of 100 documents from each of the subject classes 'astro-ph', 'cond-mat', 'gr-qc', 'hep-lat', 'hep-ph', 'hep-th', 'math-ph', 'nlin', 'quant-ph', 'physics') using a LIME or SHAP explainer can be identified using
```
get_most_discriminative_LIME.py
```
or
```
get_most_discriminative_SHAP.py
```
respectively

2) The most frequent word features are obtained using
```
get_most_frequent.py
```

3) Finally, both distributions (most discriminative, most frequent) are compared calculating distribution entropies in
```
compare_most_frequent_vs_discriminative.py
```
