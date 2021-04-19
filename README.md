# Disaster_response_project_2
Disaster response for Udacity Data Science Course

In this project I analyzed ~25k tweets that were sent after different disasters happened around the world. Tweets are classified into 36 different categories. The task was to:
1) Clean-up and prepare tweet data so it can be used in machine learning models
2) Train a machine learning model to be able to classify tweets based on words/context used
3) Finally create a web-app where users can write any message they are interested in and see how it would be classified. It also displays various graphs that summarize the most important characteristics of the data

process_data.py contains script that loads relevant tweeter data, merges it, cleans it and saves it in a new sqlite DB called DisasterResponse.

train_classifier.py loads data from DisasterResponse DB and runs AdaBoostClassifier to classify tweets in the data. Code allows to optimize parameter selection through GridSearchCV functionality. For ease of use, pre-trained model is available within the Models folder called AdaBoostClassifier_model.pkl.

![Count_per_Category](https://user-images.githubusercontent.com/61375966/115192826-9d322500-a0eb-11eb-9a34-fe602fd6b619.png)


