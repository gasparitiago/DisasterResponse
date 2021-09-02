import sys

from sqlalchemy import create_engine
import pandas as pd
import pickle
import re

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split, GridSearchCV

nltk.download(['punkt', 'wordnet', 'stopwords'])
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
stop_words = stopwords.words("english")

def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('select * from DisasterResponse', con=engine)
    X = df['message']
    Y = df.drop(['message', 'id', 'original', 'genre'], axis=1)
    return X, Y, Y.columns

def tokenize(text):
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    clean_tokens = [lemmatizer.lemmatize(word)
                    for word in tokens if word not in stop_words]
    return clean_tokens


def build_model():

    pipeline = Pipeline([('tfidf', TfidfVectorizer(tokenizer=tokenize)),
                         ('svd', TruncatedSVD(n_components=50)),
                         ('clf', MultiOutputClassifier(RandomForestClassifier()))])

    parameters = {'clf__estimator__max_depth': [50, 100],
                  'clf__estimator__min_samples_split': [2, 4],
                  'clf__estimator__n_estimators': [50, 100]}

    cv = GridSearchCV(pipeline, parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    
    Y_pred = model.predict(X_test)

    for i in range(len(category_names)):
        print('Evaluation for category: ' + category_names[i])
        print(classification_report(Y_test.iloc[:, i], Y_pred.iloc[:, i]))
        print('-----')


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()