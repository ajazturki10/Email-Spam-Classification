import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV

#Read the data
df = pd.read_csv('CSV/spam_ham.csv', encoding = 'iso-8859-1')

#Give names to Columns
df.dropna(axis = 1, inplace = True)
df.columns = ['Result', 'Reviews']

#Convert text variable in numerical 
df['Result'] = pd.get_dummies(df['Result'])

X = df['Reviews']    # Predictor Variables
y = df['Result']     # Target Variable


#Preprocessing Task
def text_process(msg):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    
    processed_data = [w for w in msg if w not in string.punctuation]
    
    #Steming process

    processed_data = [nltk.PorterStemmer().stem(w) for w in processed_data]
    processed_data = [nltk.WordNetLemmatizer().lemmatize(w) for w in processed_data]
    
    # Join the characters again to form the string.
    processed_data = ''.join(processed_data)
    
    # Now just remove any stopwords
    return [word.lower() for word in processed_data.split() if word.lower() not in stopwords.words('english')]




# Might take awhile...
vect = CountVectorizer(analyzer=text_process).fit(X)

X = vect.transform(X)


#Splitting the Dataset into Train and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

#Define the Model
spam_detect_model = MultinomialNB()

spam_detect_model.fit(X_train, y_train)


#Selecting the Best parameters for the Model
param = [{
    'alpha' : np.linspace(10e-4, 0.001, 1000)
}]

grid = GridSearchCV(spam_detect_model, param, cv = 3, scoring = 'accuracy')

grid.fit(X, y)


#select the best Model
best_nb_model = grid.best_estimator_

best_nb_model.fit(X_train, y_train)


def take_input():
    review = str(input('Type the Mail : '))
    return [review]


def classify_spam(review, take_input = False):
    if take_input == True:
        review = take_input()
        
    else:
        review = review

    rev = vect.transform(review)
    preds = best_nb_model.predict(rev)
     
    if preds == 1:
        print('Email : Ham')
        
    else:
        print('Email : Spam')
        

