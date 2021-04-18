import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
#from sklearn.externals import joblib
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
#engine = create_engine('sqlite:///https://github.com/acp91/Disaster_response_project_2/blob/main/data/DisasterResponse.db')
engine = create_engine(r'sqlite:///C:\Users\Andre\Desktop\Programming\Udacity\data_science\5\Disaster_response_project_2\data\DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("C:/Users/Andre/Desktop/Programming/Udacity/data_science/5/Disaster_response_project_2/models/AdaBoostClassifier_model.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # numbers of tweets in english vs other languages
    count_language = (df.message==df.original).value_counts().sort_values()
    count_language_cols = ['English', 'Other Languages']

    # numbers of tweets in each of the categories
    categories_summary = df.drop(columns=['message', 'original', 'genre']).sum()
    count_categories = categories_summary
    count_categories_cols = categories_summary.index

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
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
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
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
    main()