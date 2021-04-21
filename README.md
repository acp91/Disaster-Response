# Disaster_response_project_2
# Overview
Disaster response for Udacity Data Science Course

In this project I analyzed ~25k tweets that were sent after different disasters happened around the world. Tweets are classified into 35 different categories. The task was to:
1) Clean-up and prepare tweet data so it can be used in machine learning models
2) Train a machine learning model to be able to classify tweets based on words/context used
3) Finally create a web-app where users can write any message they are interested in and see how it would be classified. It also displays various graphs that summarize the most important characteristics of the data

# Folder Structure
Folder is structured as follows:
1) Data: contains underlying disaster_categories.csv and disaster_messages.csv file (this is raw underlying data). process_data.py contains script that loads both data sets, merges them, cleans the merged data set and saves it in a new sqlite DB called DisasterResponse. Folder also contains ETL_Pipeline_Preparation.ipynb jupyter notebook with step by step process on data clean-up process
2) Model: contains train_classifier.py script that loads data from DisasterResponse DB and runs AdaBoostClassifier to classify tweets in the data. Code allows to optimize parameter selection through GridSearchCV functionality. For ease of use, pre-trained model is available within the Models folder called AdaBoostClassifier_model.pkl. Folder also contains "ML)Pipeline_Preparation.ipynb jupyter notebook with step by stpe process on data modelling. For reference, this is the pipeline/gridsearch used for the classifier:

 ***
    create pipeline for the model
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('count_vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ]))
        ])),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
***
    define parameters to tweak through grid search
    parameters = {
        'features__text_pipeline__count_vect__binary': [True, False],
        'features__text_pipeline__tfidf__smooth_idf': [True, False],
        'clf__estimator__learning_rate': [1, 0.7]
    }

4) App: contains run.py script that uses above-mentioned model and flask to create a website with dashboards and ability to classify new messges/tweets. Below are some screenshot to summarize tweet data (taken from the dashboard):

Different categories:
![Count_per_Category](https://github.com/acp91/Disaster_response_project_2/blob/main/images/Count_per_Category.png)

Top 15 most popular words (excluding stopwords):
![MostPopularWords](https://github.com/acp91/Disaster_response_project_2/blob/main/images/MostPopularWords.png)

# Requirements / Packages Needed
Requirements have been created through *pip list --format=freeze > requirements.txt* command.

To load them, simply run *pip install -r requirements.txt* in the project environment.

# How to Run
1) First change directory (cd) to the folder of the project
2) To preprocess data and create your own DB, run the following command within the project directory:
> python process_data.py ".../data/disaster_messages.csv" ".../data/disaster_categories.csv" ".../data/DisasterResponse.db"

This command will preprocess the data per process_data.py file and store it in a new SQL DB called "DisasterResponse.db" within the Data folder.

3) To train your model and store it as a pickle file for later use, run the following command within the project directory:
> python train_classifier.py ".../data/DisasterResponse.db" ".../models/classifier.pkl"

This command will read data from the "DisasterResponse.db" created in the previous step, train a classifier (in this case it is AdaBoostClassifier) and store the model as a pickle file "classifier.pkl" in the "models" folder.

4) Now that you have a trained model you can view the results in the web app. To view the web app, run the following command within the "app" folder:
> python run.py

Once the dashboard is running, navigate to http://0.0.0.0:3001/ to access it. You can type any message (but makes most sense to use a message that relates to a disaster/emergency) and it will be classified according to pre-trained model from step 3).

For example, message "The earthquake in the nearby village destroyed many buildings. Many injured people were taken to the hospital" would be classified as Aid Related, Infastructure Related, Hospitals, Weather Related and Earthquake.

![Classified_message](https://github.com/acp91/Disaster_response_project_2/blob/main/images/Classified_message.png)

# Model Evaluation / Comments on Performance
train_classifier.py script also prints out classification report for each of the message categories. Please see Acknowledgements section for links with clear explanation on how to interpret accuracy, precision, recall and F1.

Accuracy, defined as <img src="https://render.githubusercontent.com/render/math?math=\large \frac{TruePositive %2B TrueNegative}{TruePositive %2B TrueNegative %2B FalsePositive %2B FalseNegative}"> is only useful for set that has balanced data (e.g. number of positive and negative outcomes is proporationte). This is not the case for disaster response messages so let's rather look at the other 3 metrics on below example for "request" category:

![classification_request](https://github.com/acp91/Disaster_response_project_2/blob/main/images/classification_request.png)

Precision is defined as <img src="https://render.githubusercontent.com/render/math?math=\large \frac{TruePositive}{TruePositive %2B FalsePositive}">. For "request" it is as 0.78 which means 78% of messages, classified as "request" were correctly classified (i.e. they were indeed related to "request"). Precision is good for measuring your model's performance when the cost of false positive is high. False positive in our case would be message classified as "request" when in fact it isn't. We can say that the model performs relatively well in identifying false positives.

Recall on the other hand is defined as <img src="https://render.githubusercontent.com/render/math?math=\large \frac{TruePositive}{TruePositive %2B FalseNegative}">. For "request" it is at 0.31 which means the model correctly classified as "request" 31% of all messages that relate to "request" . Recall is good for measuring your model's performance when the cost of false negative is high. False negative in our case would be when message isn't classified as "request" but in fact it refers to a request. We can say that the model performs poorly in identifying false negatives.

It's probably fair to assume that in case of disaster response, recall is the more relevant measure - if there is indeed an emergency, we'd want to know it asap so we can respond accordingly. Therefore we can conclude that for "request", the model is not very useful.

We could look further into F1 score, which combines both metrics, but for now let's focus on recall only as we concluded is the more relevant metric. Let's look at some of the other categories where people need emergency help:

* Medical Help:

![medical_help](https://github.com/acp91/Disaster_response_project_2/blob/main/images/medical_help.png)

* Food:

![food](https://github.com/acp91/Disaster_response_project_2/blob/main/images/food.png)

* Earthquake:

![earthquake](https://github.com/acp91/Disaster_response_project_2/blob/main/images/earthquake.png)

What do above images tell us? Based on recall, we have to be careful what model predictions to trust. For some classifications it is much more useful/reliable than others. "Medical help" has a low recall of 19%, "food" of 55% and "earthquake" of 75%.

One possible reason for that is the way different messages are usually phrased by people and shared. When referring to earthquake, messages would be mostly unambiguous as earthquake is specific event with consequences such as collapsed buildings. Asking for medical help on the other hand can be phrased in many differnt ways and can refer to many different scenarios and severities.

# Acknowledgements
* [Precision, Recall - developers.google](https://developers.google.com/machine-learning/crash-course/classification/precision-and-recall)
* [Accuracy - developers.google](https://developers.google.com/machine-learning/crash-course/classification/accuracy)
* [Accuracy, Precision, Recall, F1 - towardsdatascienec](https://towardsdatascience.com/accuracy-precision-recall-or-f1-331fb37c5cb9)
* [Docstrings in Python](https://www.datacamp.com/community/tutorials/docstrings-python)
* [Github Readme](https://github.com/matiassingers/awesome-readme)
* [Mastering Github Markdown](https://guides.github.com/features/mastering-markdown/)
* [LaTeX in Github](https://gist.github.com/a-rodin/fef3f543412d6e1ec5b6cf55bf197d7b)