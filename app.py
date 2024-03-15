from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from tensorflow.keras.preprocessing.text import Tokenizer
import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

app = Flask(__name__)

def Cleaning(text):
    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
    BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
    STOPWORDS = set(stopwords.words('english'))
    MAX_NB_WORDS = 50000

    MAX_SEQUENCE_LENGTH = 250
  
    EMBEDDING_DIM = 100
    text = text.lower()
    text = REPLACE_BY_SPACE_RE.sub(' ', text) 
    text = BAD_SYMBOLS_RE.sub('', text) 
    
   
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) 
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    tokenizer.fit_on_texts(text)
    word_index = tokenizer.word_index
    text= tokenizer.texts_to_sequences(text)
    text= pad_sequences(text, maxlen=MAX_SEQUENCE_LENGTH)


    return text


@app.route('/', methods=['GET', 'POST'])
def index():
    category = None
    if request.method == 'POST':
        text = request.form['text']
        text=Cleaning(text)
        
        models=load_model(r'C:\Users\HP\Desktop\Kaviya\Hack\Email_Classification_LSTM_1.hdf5')
        y_pred=models.predict(text)

        print("hello",y_pred)
        predicted_classes = np.argmax(y_pred, axis=1)

        print("output",predicted_classes)

        label_mapping=label_mapping = {
                                    2: 'Credit & Debit Cards',
                                    3: 'Credit Reports & Identity Protection',
                                    5: 'Debt Recovery & Collections',
                                    0: 'Banking & Savings Accounts',
                                    8: 'Home Loans & Mortgages',
                                    13: 'Vehicle Financing & Leasing',
                                    7: 'Money Transfer & Virtual Currency Services',
                                    12: 'Student Loans & Education Financing',
                                    11: 'Prepaid Card Services',
                                    10: 'Personal Loans & Cash Advances',
                                    6: 'Financial Management & Counseling',
                                    4: 'Credit Repair & Improvement Services',
                                    9: 'Consumer Loans & Financing',
                                    1: 'Other Financial Services'  # Adding a catch-all category for any remaining labels
                                }

        predicted_categories = [label_mapping[label] for label in predicted_classes]

        print("category",predicted_categories)
        category = predicted_categories[0]

    return render_template('index.html', category=category)

if __name__ == '__main__':
    app.run(debug=True)
