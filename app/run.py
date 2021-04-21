import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
# from sklearn.externals import joblib
import joblib
from sqlalchemy import create_engine

import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import re

app = Flask(__name__)


def tokenize(text):
    """
    Prepares input string by lemmatizing, tokenizing and normalizing the text.

    :param text: string to be tokenized
    :return: list of tokenized text
    """

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# another tokenize function -> will be used to count most popular words used in tweets
def tokenize_graph(text):
    """
    Prepares input string by removing stop words, punctuations, reducing words to their root forms, normalizing and tokenizing the text.

    :param text: string to be tokenized
    :return: list of tokenized text
    """

    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()

    # normalize text
    text = text.lower()
    # remove punctuation
    text = re.sub(r'[^a-zA-A0-9]', ' ', text)
    # tokenize the text
    text = word_tokenize(text)
    # Remove stop words
    text = [w for w in text if w not in stop_words]

    # Lemmatization
    # Reduce words to their root form
    # lemmatize for nouns
    text = [lemmatizer.lemmatize(x) for x in text]
    # lemmatize for verbs
    text = [lemmatizer.lemmatize(x, pos='v') for x in text]

    return text


def main_words(df):
    """
    Counts the number of times each word is used

    :param df: df object with strings
    :return: df object with 2 columns: words used in sentences and count of how many times they were used
    """

    # get dictionary of most popular words
    all_words = dict()
    nr_tweets = df.shape[0]
    for x in range(nr_tweets):
        sentence = tokenize_graph(df.message[x])
        for word in sentence:
            if word not in all_words:
                all_words[word] = 0
        all_words[word] += 1

    # create dataframe from dictionary
    all_words = pd.DataFrame(all_words, index=[0]).transpose()
    all_words = all_words.reset_index()
    # sort the dataframe
    all_words.sort_values(by=0, ascending=False, inplace=True)

    return all_words


# load data
# engine = create_engine('sqlite:///https://github.com/acp91/Disaster_response_project_2/blob/main/data/DisasterResponse.db')
#engine = create_engine(r'sqlite:///C:\Users\Andre\Desktop\Programming\Udacity\data_science\5\Disaster_response_project_2\data\DisasterResponse.db')
#df = pd.read_sql_table('DisasterResponse', engine)
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
#model = joblib.load("C:/Users/Andre/Desktop/Programming/Udacity/data_science/5/Disaster_response_project_2/models/classifier.pkl")
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    """
    add visuals (4 different bar charts)

    :return: rendered template with encoded plotly graphs in JSON
    """

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # numbers of tweets in english vs other languages
    count_language = (df.message == df.original).value_counts().sort_values()
    count_language_cols = ['English', 'Other Languages']

    # numbers of tweets in each of the categories
    categories_summary = df.drop(columns=['message', 'original', 'genre']).sum().sort_values(ascending=False)
    count_categories = categories_summary
    count_categories_cols = categories_summary.index

    # count most popular words (excluding stop words)
    all_words = main_words(df)
    popular_words_names = all_words.head(15)['index']
    popular_words_cols = all_words.head(15)[0]

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=count_categories_cols,
                    y=count_categories
                )
            ],

            'layout': {
                'title': 'Count of Messages per Category',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=popular_words_names,
                    y=popular_words_cols
                )
            ],

            'layout': {
                'title': 'Most Popular Words in Tweets',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Words"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=count_language_cols,
                    y=count_language
                )
            ],

            'layout': {
                'title': 'Distribution of Messages in Languages',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Languages"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    """
    Input field where user can write free text. Message will be classified based on the pre-trained model

    :return: rendered template with input box to classify message
    """

    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    #main()