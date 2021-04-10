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
    # 1) this function loads both messages and categories data sets
    # 2) it then merges them together in a one data set
    # 3) afterwards it extracts relevant category names and values from categories dataset and use them as 
    # the new column names / values for categories
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, left_index=True, right_index=True)
    
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
    # drop duplicate values
    df.drop_duplicates(inplace=True)


def save_data(df, database_filename):
    # create a new DB engine
    engine = create_engine('sqlite:///DisasterResponse.db')
    df.to_sql('DisasterResponse', engine, index=False)

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()