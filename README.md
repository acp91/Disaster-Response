# Disaster_response_project_2
# Overview
Disaster response for Udacity Data Science Course

In this project I analyzed ~25k tweets that were sent after different disasters happened around the world. Tweets are classified into 36 different categories. The task was to:
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
![Count_per_Category](https://user-images.githubusercontent.com/61375966/115192826-9d322500-a0eb-11eb-9a34-fe602fd6b619.png)

Top 15 most popular words (excluding stopwords):
![MostPopularWords](https://user-images.githubusercontent.com/61375966/115208971-6fee7280-a0fd-11eb-80d3-5b67dda8f09e.png)

# How to Run
1) First change directory (cd) to the folder of the project, e.g. "acp91/Disaster_response_project_2"
2) To preprocess data and create your own DB, run the following command within the project directory:
> python process_data.py ".../data/disaster_messages.csv" ".../data/disaster_categories.csv" ".../data/DisasterResponse.db"

This command will preprocess the data per process_data.py file and store it in a new SQL DB called "DisasterResponse.db" within the Data folder.

3) To train your model and store it as a pickle file for later use, run the following command within the project directory:
> python train_classifier.py ".../data/DisasterResponse.db" ".../models/classifier.pkl"

This command will read data from the "DisasterResponse.db" created in the previous step, train a classifier (in this case it is AdaBoostClassifier) and store the model as a pickle file "classifier.pkl" in the "models" folder.

4) Now that you have a trained model you can view the results in the web app. To view the web app, run the following command within the "app" folder:
> python run.py

Once the dashboard is running, navigate to http://0.0.0.0:3001/ to access it.
