from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import re, string, os, pickle

CUR_DIR = os.path.dirname(__file__)
STOP_WORDS = pickle.load(open(
                os.path.join(CUR_DIR,
                'pkl_objects',
                'stopwords.pkl'), 'rb'))
VECTORIZER = pickle.load(open(
                os.path.join(CUR_DIR,
                'pkl_objects',
                'vectorizer.pkl'), 'rb'))
CLF = pickle.load(open(
                os.path.join(CUR_DIR,
                'pkl_objects',
                'classifier.pkl'), 'rb'))

LABEL_DICT = {0: 'The tweet contains hate speech', 
              1: 'The tweet is not offensive', 
			  2: 'The tweet uses offensive language but not hate speech'}

app = Flask(__name__)

class ReviewForm(Form):
    tweet_classifier = TextAreaField('',
                                [validators.DataRequired(),
                                validators.length(min = 5)])

@app.route('/')
def index():
    form = ReviewForm(request.form)
    return render_template('submission.html', form = form)

@app.route('/results', methods = ['POST'])
def results():
    form = ReviewForm(request.form)
    if request.method == 'POST' and form.validate():
        tweet = request.form['tweet_classifier']
        y, proba = classify_tweet(tweet)
        return render_template('results.html',
                                content = tweet,
                                prediction = y,
                                probability = proba)
    return render_template('submission.html', form = form)

@app.route('/about')
def about():
    return render_template('about.html')

def processTweet(tweet):

    tweet = tweet.lower()    
    #Remove urls
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', '', tweet)    
    #Remove usernames
    tweet = re.sub('@[^\s]+','',tweet)    
    #Remove white space
    tweet = tweet.strip()    
    #Remove hashtags
    tweet = re.sub(r'#([^\s]+)', '', tweet)   
    #Remove stopwords
    tweet = " ".join([word for word in tweet.split(' ') if word not in STOP_WORDS])
    #Remove punctuation
    tweet = "".join(l for l in tweet if l not in string.punctuation)
    
    return tweet

def classify_tweet(tweet):
    tweet_to_clf = processTweet(tweet)
    tweet_to_clf = VECTORIZER.transform([tweet_to_clf])
    label = CLF.predict(tweet_to_clf)[0]
    confidence = max(CLF.predict_proba(tweet_to_clf)[0])*100
    return LABEL_DICT[label], round(confidence, 2)

if __name__ == '__main__':
    app.run(debug = True)