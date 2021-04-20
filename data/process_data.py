# final command to run: python process_data.py 'https://github.com/acp91/Disaster_response_project_2/blob/main/data/disaster_messages.csv?raw=true' 'https://github.com/acp91/Disaster_response_project_2/blob/main/data/disaster_categories.csv?raw=true' 'https://github.com/acp91/Disaster_response_project_2/blob/main/data/DisasterResponse.db'


# import libraries
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import sqlite3


def load_data(messages_filepath, categories_filepath):
    """
    Loads messages and categories data sets,  merges and cleans them and returns a new DataFrame

    :param messages_filepath: path for the messages dataset (csv format)
    :param categories_filepath: path for the categories dataset (csv format)
    :return: df: pandas DataFrame
    """

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, left_index=True, right_index=True)
    # drop index columns for both datasets
    df.drop(['id_x', 'id_y'], axis=1, inplace=True)

    # create a dataframe of individual category columns
    categories = categories['categories'].str.split(pat=';', expand=True)
    # use first row to extract lsit of new column names for categories
    category_colnames = categories.iloc[0].apply(lambda x: x[:-2])
    # rename the columns of categories
    categories.columns = category_colnames

    # Iterate through the category columns in df to keep only the last character of each string (the 1 or 0)
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype('int')

        # drop the original categories column from `df`
    df.drop(columns=['categories'], inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    return df


def clean_data(df):
    """

    :param df: df object
    :return: df after duplicates are dropped
    """

    # drop duplicate values
    df.drop_duplicates(inplace=True)
    return df


def save_data(df, database_filename):
    """
    Save df to SQL database per specified filepath

    :param df: df object
    :param database_filename: output path where SQL databse is saved to
    :return: -
    """

    # create a new DB engine
    engine = create_engine('sqlite:///DisasterResponse.db')
    df.to_sql(database_filename, engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)
        print(df.head())
        print(df.shape)

        print('Cleaning data...')
        df = clean_data(df)
        print(df.head())
        print(df.shape)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
