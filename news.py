#Importing  required libraries 
import requests
from bs4 import BeautifulSoup
import numpy as np  
import pandas as pd
import re, nltk
from nltk.tokenize import WordPunctTokenizer
tokenizer = WordPunctTokenizer()
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer,HashingVectorizer
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix as cfm
from sklearn.metrics import f1_score

#Scraping news headlines from india_today using beatifulsoup
def get_news(url, no_of_pages):
  news = []
  for i in range(no_of_pages):
    url = url + str(i)
    page = requests.get(url).text
    soup = BeautifulSoup(page)
    a_tags = soup.find_all('a')
    a_tags_text = [tag.get_text().strip() for tag in a_tags]
    sentence_list = [sentence for sentence in a_tags_text if len(sentence)>25]
    news = news + sentence_list[:12]
  return news

number_of_pages=20
trending_url = 'https://www.indiatoday.in/trending-news?page='
normal_url   = 'https://www.indiatoday.in/world?page='

x_trend  = get_news(trending_url,number_of_pages)
x_normal = get_news(normal_url,number_of_pages)

print('Before Processing')
print()
print(x_trend[:5])

#getting today's headlines to predict their virality likelihood
test_news = get_news('https://www.indiatoday.in/mail-today?page=', 10)


labels=[1]*len(x_trend) + [0]*len(x_normal)
train_data=x_trend + x_normal


#Cleaning text data through removing punctuation,stopwords & using lemmatisation 
stop_words = set(stopwords.words('english'))
wordnet_lemmatizer = WordNetLemmatizer()
def data_cleaning(news):
    proc=[]
    for line in news:
        only_letters = re.sub("[^a-zA-Z0-9]", " ",line) 
        tokens = nltk.word_tokenize(only_letters)
        lower_case = [l.lower() for l in tokens]
        filtered_result = list(filter(lambda l: l not in stop_words, lower_case))
        lemmas = ' '.join([wordnet_lemmatizer.lemmatize(t) for t in filtered_result])
        proc.append(lemmas)
    return proc        

cleaned_training = data_cleaning(train_data)
cleaned_test     = data_cleaning(test_news)

print('no. of data points for training: ',len(cleaned_training))
print('no. of data points predicted:    ',len(cleaned_test))

#Using hashing vectorising method for word embedding representation
vectorizer = HashingVectorizer(n_features=500)
train_vectorized = vectorizer.fit_transform(cleaned_training)
test_vectorised  = vectorizer.transform(cleaned_test)

#Splitting data into train & validation
x_train,x_val,y_train,y_val=train_test_split(train_vectorized,labels,test_size=0.25,shuffle=True,stratify=labels,random_state=5)

print("number of training data points:   ",x_train.shape[0])
print('number of validation data points: ',x_val.shape[0])

#Building and Fitting a simple Logistic Regression Model on training data
log_reg=LogisticRegression()
log_reg.fit(x_train,y_train)

#Results of Logistic Regression model
y_pred=log_reg.predict(x_val)
print('accuracy: ',log_reg.score(x_val,y_val))
print('f1_score: ',f1_score(y_val,y_pred))
print('confusion matrix: ')
print(cfm(y_val,y_pred))

#Using the trained regression model for predicting likelihood score on test data
prediction=log_reg.predict_proba(test_vectorised)
model_predictions=pd.DataFrame({'news':test_news,'virality_likelihood':prediction[:,1]*100})
model_predictions.head()

#Saving predictions to csv file
model_predictions.to_csv('predicted_virality.csv')
