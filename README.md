# Disaster_response_project_2
Disaster response for Udacity Data Science Course

process_data.py contains script that loads relevant tweeter data, merges it, cleans it and saves it in a new sqlite DB called DisasterResponse.

train_classifier.py loads data from DisasterResponse DB and runs AdaBoostClassifier to classify tweets in the data. Code allows to optimize parameter selection through GridSearchCV functionality. For ease of use, pre-trained model is available within the Models folder called AdaBoostClassifier_model.pkl.

