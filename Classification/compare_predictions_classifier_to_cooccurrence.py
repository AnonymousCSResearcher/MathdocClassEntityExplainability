import json

# load predictions classifier and coocurrence
with open("predictions_classifier.json",'r') as f:
    predictions_classifier = json.load(f)
with open("predictions_coocurrence.json",'r') as f:
    predictions_coocurrence = json.load(f)

# compare predictions
comparisons = []
for prediction in predictions_classifier.items():
    predictor = prediction[0]
    prediction_classifier = prediction[1]
    prediction_coocurrence = predictions_coocurrence[predictor]
    comparisons.append((prediction_classifier,prediction_coocurrence))
    print(int(prediction_classifier) == int(prediction_coocurrence))

print("end")