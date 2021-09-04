# Disaster Response Pipeline Project

## Installation
Just install the dependencies using pip:

```
pip install -r requirements.txt
```

## Project Overview

The main objective is to create a system responsible of classifying disaster messages. For that, I'm using a database from [Figure Eight](https://www.figure-eight.com/) containing real messages that were sent during disaster events.

The project code is designed to put into practice different aspects of Data Science, including:
- Usage of an ETL pipeline to read different csv files, to extract, tranform and load the data, creating a sql database.
- NLP (Natural Language Processing) techniques, including tokenizer and lemmatizer to create a Machine Learning model to categorize messages into different categories.
- Creating a Flask app that uses the database created to visualize the data and the machine learning model to perform inferences: users can type messages that are categorized.

This project is part of the Data Science Nanodegree from [Udacity](https://udacity.com/).

## Files Description:

* **data/process_data.py**: Performs the ETL pipeline: takes as input csv files containing message data and message categories (labels), transform the data and store it in a SQL database.

* **models/train_classifier.py**: Using NLP techniques, create tokens and train a machine learning model to classify the messages.

* **app/run**: Code to iniate the Flask Web App.

### Instructions:
1. Run the commands in the project's root directory to create the SQL database and the classification model.

    - ETL Pipeline:
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

    - ML Pipeline:
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/ to use the Flask Web App.

## Screenshots

***Screenshot: App: Visualization of the database***
![Screenshot_1](https://raw.githubusercontent.com/gasparitiago/DisasterResponse/main/screenshots/Screenshot1.png)

***Screenshot 2: App: Model inference used to classify messages***
![Screenshot_2](https://raw.githubusercontent.com/gasparitiago/DisasterResponse/main/screenshots/Screenshot2.png)


## Licensing, Authors, Acknowledgements

Author: Tiago De Gaspari.

License: The code provided is under BSD 3-Clause License.

Acknowledgement: [Udacity](https://udacity.com/)
