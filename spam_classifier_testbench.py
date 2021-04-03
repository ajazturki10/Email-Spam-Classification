import joblib

#load the saved model
model = joblib.load("spam_classifier_model.pkl")



def take_input():
    review = str(input('Type the Mail : '))
    return [review]


def classify_spam(review, take_input = False):
    if take_input == True:
        review = take_input()
        
    else:
        review = review

    rev = vect.transform(review)
    preds = model.predict(rev)
     
    if preds == 1:
        print('Email : Ham')
        
    else:
        print('Email : Spam')
