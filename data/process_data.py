import sys

import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load the data from messages and categories into a pandas DataFrame

    Args:
    messages_filepath -- path of csv file containing the messages
    categories_filepath -- path of csv file containing the categories of the messages
    
    Returns:
    pandas dataframe containing messages and categories
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # Merge both messages and categories using the 'id' column
    df = pd.merge(messages, categories, left_on='id', right_on='id', how='left')
    return df


def clean_data(df):
    """
    Clean the dataframe creating columns for each one of the labels
    based on the different categories and removing duplicated messages.

    Args:
    df -- dataframe containing messages and categories
    
    Returns:
    cleaned dataframe without duplicated and with new columns
    based on the categories labels. 
    """
    # Split the categories by the ';' symbol
    categories = pd.Series(df['categories']).str.split(';', expand=True)
    
    # Gets the first line to use as the columns names
    # removing the '-0' or '-1' from the end of the string
    row = categories.iloc(0)[0]
    category_colnames = list(row)
    for i in range(len(category_colnames)):
        category_colnames[i] = category_colnames[i][:-2]

    categories.columns = category_colnames
    
    # For each column created, replace the values by just the last character
    # in order to keep only 0 or 1 in each column.
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # Remove rows that have '2' on any of the columns
    categories = categories[~(categories == 2).any(axis=1)]

    # Remove the original categories column and replace with the new columns
    df.drop(columns=['categories'], inplace=True)
    df = df.join(categories)
    df.drop_duplicates(inplace=True)

    return df
    

def save_data(df, database_filename):
    """
    Save the data from a pandas dataframe into a sqlite database.
  
    Args:
    df - Input database.
    database_filename - filename of the output sqlit database file.
    """
    # Create the enfine using sqlalchemy package
    engine = create_engine('sqlite:///' + database_filename)
    # Convert dataframe to sql
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')


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