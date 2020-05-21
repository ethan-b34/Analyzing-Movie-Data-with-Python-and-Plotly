import numpy as np
import pandas as pd
pd.set_option('max_columns', None)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
plt.style.use('ggplot')
import datetime
import lightgbm as lgb
from scipy import stats
from scipy.sparse import hstack, csr_matric
from sklearn.model_selection import train_test_split, KFold
from wordcloud import WordCloud
from collections import Counter
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidVectorizer, CountVectorizer
from sklearn.preprocessing import Standard Scaler
import nltk
nltk.download('stopwords')
stop = set(stopwords.words('english'))
import os
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import xgboost as xgb
import lightgbm as lgb
from sklearn import model_selection
from sklearn.metrics import accuracy_score
import json
import ast
from urllib.request import urlopen
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import time
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegresssion
from sklearn import liner_model


train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
train.head()



fig, ax = plt.subplots(figsize = (16, 6))
plt.subplot(1, 2, 1)
plt.hist(train['revenue']);
plt.title('Distribution of revenue');
plt.subplot(1, 2, 2)
plt.hist(np.loglp(train['revenue']));
plt.title('Distribution of log of revenue');


train['log_revenue'] = np.loglp(train['revenue'])


fig, ax = plt.subplots(figsize = (16, 6))
plt.subplot(1, 2, 1)
plt.hist(train['budget']);
plt.title('Distribution of budget');
plt.subplot(1, 2, 2)
plt.hist(np.loglp(train['budget']));
plt.title('Distribution of log of budget');


plt.figure(figsize = (16, 8))
plt.subplot(1, 2, 1)
plt.scatter(train['budget'], train['revenue']);
plt.title('Revenue vs Budget');
plt.subplot(1, 2, 2)
plt.hist(np.loglp(train['budget']), train['log_revenue']);
plt.title('Log Revenue vs log budget');


train['log_budget'] = np.loglp(train['budget'])
test['log_budget'] = np.loglp(test['budget'])


train['homepage'].value_counts().head(10)

train['has_homepage'] = 0
train.loc[train['homepage'].isnull() == False, 'has_homepage'] = 1
test['has_homepage'] = 0
test.loc[test['homepage'].isnull() == False, 'has_homepage'] = 1


sns.catplot(x='has_homepage', y='revenue', data=train);
plt.title('Revenue for film with and without homepage');

#Distribution of Languages in Film
plt.figure(figsize=(16,8))
plt.subplot(1, 2, 1)
sns.boxplot(x='original_language', y='revenue', data=train.loc[train['original_language']isin(train['original_language'].value_counts().head(10).index)]);
plt.title('Mean revenue per language');
plt.subplot(1, 2, 2)
sns.boxplot(x='original_language', y='log_revenue', data=train.loc[train['original_language']isin(train['original_language'].value_counts().head(10).index)]);
plt.title('Mean log revenue per language');


#Do film descriptions impact revenuw?
import eli5

vectorizer = TfidVectorizer(
            sublinear_tf=True,
            analyzer='word'
            token_pattern=r'\w{1,}'
            ngram_range=(1,2),
            min_df=5)

overview_text = vectorizer.fit_transform(train['overview'].fillna(''))
linreg = LinearRegression()
linreg.fit(overview_text, train['log_revenue'])
eli5.show_weights(linreg, vec=vectorizer, top=20, feature_filter=lambda x: x != '<BIAS>')

print('Target value:', train['log_revenue'][1000])
eli5.show_prediction(linreg, doc=train['overview'].values[1000], vec=vectorizer)


    








                                                                                                      




































