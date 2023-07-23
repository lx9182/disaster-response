import sys
from sqlalchemy import create_engine
import re
import pandas as pd
import nltk
import pickle

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier


nltk.download('punkt')
nltk.download('stopwords')

def load_data(database_filepath):
    '''
    Load data from database
    '''
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql('select * from disaster_response', engine)
    X = df['message']
    y = df.iloc[:, 4:]

    # From EDA
    # `child_alone` contains only value 0, it should be dropped
    # `related` has some value 2, replace with 1 because value 1 is more frequent then value 0
    y.drop('child_alone', axis=1, inplace=True)
    y['related'] = y['related'].map(lambda x : 1 if x == 2 else x)

    # Get category list
    category_names = list(y.columns)

    return X, y, category_names


def tokenize(text):
    '''
    Tokenize word before fit it to model
    '''
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    normlizer = PorterStemmer()
    stop_words = stopwords.words("english")
    normlized = [normlizer.stem(word) for word in tokens if word not in stop_words]
    
    return normlized


def build_model():
    '''
    Create model pipeline
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evalute model
    '''
    y_pred = model.predict(X_test)
    print(classification_report(Y_test.values, y_pred, target_names=category_names))


def save_model(model, model_filepath):
    '''
    Save model
    '''
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