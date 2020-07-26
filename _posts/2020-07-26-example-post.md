---
title: "Identification of Twitter Communities Interested in African Affairs"
date: 2020-07-26
tags: [data wrangling, data science, messy data]
header:
  image: "/images/perceptron/percept.jpg"
excerpt: "Data Wrangling, Data Science, Messy Data,Twitter,Data Minining"
mathjax: "true"
---

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

```


```python

import pyLDAvis
import pyLDAvis.gensim  # 
import matplotlib.pyplot as plt
```


```python
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
```


```python
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
```


```python
import random
import re
import os
import json
import time
import tweepy
import jsonpickle
import preprocessor as p
import nltk
from nltk.corpus import stopwords
import string
from textblob import TextBlob
import matplotlib.pyplot as plt
import numpy as np
#nltk.download('punkt')

from requests import get
from requests.exceptions import RequestException
from contextlib import closing
from bs4 import BeautifulSoup
import pandas as pd
import os, sys
import re
import fire
from collections import Counter
import requests
```


```python
import logging
import os
import pandas as pd
import re
import scrapy
from googlesearch import search
from scrapy.crawler import CrawlerProcess
from scrapy.linkextractors.lxmlhtml import LxmlLinkExtractor
```


```python
from textblob import TextBlob

#general text pre-processor
#!pip install nltk
import nltk
from nltk.corpus import stopwords
#nltk.download('punkt')

#tweet pre-processor 
#!pip install tweet-preprocessor
import preprocessor as p
import numpy as np

```


```python
import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy
```

    C:\Users\Hp\Anaconda3\lib\site-packages\gensim\utils.py:1197: UserWarning: detected Windows; aliasing chunkize to chunkize_serial
      warnings.warn("detected Windows; aliasing chunkize to chunkize_serial")
    


```python
#%%writefile ../pyscrap_url.py

def simple_get(url):
    """
    Attempts to get the content at `url` by making an HTTP GET request.
    If the content-type of response is some kind of HTML/XML, return the
    text content, otherwise return None.
    """
    try:
        with closing(get(url, stream=True)) as resp:
            if is_good_response(resp):
                return resp.content  #.encode(BeautifulSoup.original_encoding)
            else:
                return None

    except RequestException as e:
        log_error('Error during requests to {0} : {1}'.format(url, str(e)))
        return None


def is_good_response(resp):
    """
    Returns True if the response seems to be HTML, False otherwise.
    """
    content_type = resp.headers['Content-Type'].lower()
    return (resp.status_code == 200 
            and content_type is not None 
            and content_type.find('html') > -1)


def log_error(e):
    """
    It is always a good idea to log errors. 
    This function just prints them, but you can
    make it do anything.
    """
    print(e)
    
def get_elements(url, tag='h2',search={}, fname=None):
    """
    Downloads a page specified by the url parameter
    and returns a list of strings, one per tag element
    """
    
    if isinstance(url,str):
        response = simple_get(url)
    else:
        #if already it is a loaded html page
        response = url

    if response is not None:
        html = BeautifulSoup(response, 'html.parser')
        
        res = []
        if tag:    
            for li in html.select(tag):
                for name in li.text.split('\n'):
                    if len(name) > 0:
                        res.append(name.strip())
                       
                
        if search:
            soup = html            
            
            
            r = ''
            if 'find' in search.keys():
                print('findaing',search['find'])
                soup = soup.find(**search['find'])
                r = soup

                
            if 'find_all' in search.keys():
                print('findaing all of',search['find_all'])
                r = soup.find_all(**search['find_all'])
   
            if r:
                for x in list(r):
                    if len(x) > 0:
                        res.extend(x)
            
        return res

    # Raise an exception if we failed to get any data from the url
    raise Exception('Error retrieving contents at {}'.format(url))    
    
    
if get_ipython().__class__.__name__ == '__main__':
    fire(get_elements)
```


```python
def get_urls(tag,n,language):
    urls=[url for url in search(tag,stop=n,lang=language,pause=2.0)][:n]
    return urls
          
```


```python
url_journalist=get_urls('Twitter handles of top journalist in Nigeria',100,'en')
```


```python
url_government=get_urls('Twitter handles of government officials in Nigeria',100,'en')

```


```python
url_celebrities=get_urls('Twitter handles of celebrities in Nigeria',100,'en')
```


```python
url_celebrities
```


```python
for url in url_celebrities:
    if 'twitter.com' in url:
        print (url)
```


```python
##(@\w+)|
a=re.search('(?:https?:\/\/)?(?:www\.)?twitter\.com\/(?:#!\/)?@?([^\/\?\s]*)',str(url_government[1]))
print (a[1])
```


```python
journalist=[]
for i in url_journalist:
    search_=re.search('(?:https?:\/\/)?(?:www\.)?twitter\.com\/(?:#!\/)?@?([^\/\?\s]*)',str(i))
    if search_ !=None:
        journalist.append(search_[1])
```


```python
govt_official=[]
for i in url_government:
    search1=re.search('(?:https?:\/\/)?(?:www\.)?twitter\.com\/(?:#!\/)?@?([^\/\?\s]*)',str(i))
    if search1 !=None:
        govt_official.append(search1[1])
```


```python
govt_official
```


```python
url= 'https://www.nairaland.com/1166340/naija-celebrities-twitter-handles'
response = simple_get(url)

res = get_elements(response, search={'find_all':{'class_':'narrow'}})
res
```

    findaing all of {'class_': 'narrow'}
    




    ['Naija Celebrities Twitter Handles - Entertainment - Nairaland',
     "Here is a list of some of Nigeria's celebrities twitter handle",
     <br/>,
     <br/>,
     'Wizkid = @Wizkidayo',
     <br/>,
     'Ice Prince = @Iceprincezamani',
     <br/>,
     'Banky W = @BankyW',
     <br/>,
     'Cossy = @Cossydiva',
     <br/>,
     'Tonto Dikeh = @TONTOLET',
     <br/>,
     'Olamide = @olamide_YBNL',
     <br/>,
     'Don Jazzy = @DONJAZZY',
     <br/>,
     "D'banj = @iamdbanj",
     <br/>,
     'Tiwa Savage = @TiwaSavage',
     <br/>,
     'Omotola = @Realomosexy',
     <br/>,
     'Rita Dominic = @ritaUdominic',
     <br/>,
     'Uche Jombo = @uchejombo',
     <br/>,
     'Davido = @iam_Davido',
     <br/>,
     'Phyno = @phynofino',
     <br/>,
     'Wande Coal = @wandecoal',
     <br/>,
     <br/>,
     <br/>,
     'And finally my own @RealStaceyBanks or ',
     <a href="http://www.twitter.com/realstaceybanks">www.twitter.com/realstaceybanks</a>,
     <br/>,
     <br/>,
     <br/>,
     'Thank You',
     ' ',
     <iframe allowfullscreen="" class="youtube" frameborder="0" src="https://www.youtube.com/embed/gEAfdlpKVQw"></iframe>,
     <br/>,
     <a href="https://www.youtube.com/watch?v=gEAfdlpKVQw">https://www.youtube.com/watch?v=gEAfdlpKVQw</a>,
     ' ',
     <iframe allowfullscreen="" class="youtube" frameborder="0" src="https://www.youtube.com/embed/avmMplwo688"></iframe>,
     <br/>,
     <a href="https://www.youtube.com/watch?v=avmMplwo688">https://www.youtube.com/watch?v=avmMplwo688</a>,
     ' ',
     <iframe allowfullscreen="" class="youtube" frameborder="0" src="https://www.youtube.com/embed/Erx2fqnZ7xI"></iframe>,
     <br/>,
     <a href="https://www.youtube.com/watch?v=Erx2fqnZ7xI">https://www.youtube.com/watch?v=Erx2fqnZ7xI</a>,
     ' ',
     <iframe allowfullscreen="" class="youtube" frameborder="0" src="https://www.youtube.com/embed/l2kZIkSK5Cw"></iframe>,
     <br/>,
     <a href="https://www.youtube.com/watch?v=l2kZIkSK5Cw">https://www.youtube.com/watch?v=l2kZIkSK5Cw</a>,
     'BECOME TECTONO LADY OF THE WEEK',
     <br/>,
     'Tectono Business Review is the Nigerian version of Harvard Business Review, Oxford Business Group, Financial Times, The Economist, EuroMoney etc. It publishes business news, business articles and adverts. The main people that read our publications several times in a day are the elites such as business tycoons, top politicians, captains of industries, heads of federal parastatals, CEOs and top managers of multinational companies. We publish via our website: ',
     <a href="http://www.tectono-business.com/">www.tectono-business.com/</a>,
     ' We would like you to be the face of our publishing firm for the week? It is a weekly contest.',
     <br/>,
     <br/>,
     'So, do you consider yourself beautiful, educated, intelligent and worthy of being published in the most sought after business consultancy, research and publishing website worldwide? Do you want your profession, skills and all your good qualities to be showcased all over the world? Then, apply to become Tectono Lady of the Week.',
     <br/>,
     <br/>,
     'Yes, anyone who wins Tectono Lady of the Week will have her picture, phone number (optional), profession and a full description of herself published in Tectono Business Review and shared on all social media platforms. Do you know what it means for millions of people to know you, what you do and what you intend to do in the future and also contact you? It is the easiest avenue of becoming a superstar in your own field of endeavor and also prosperous. To apply, click the link below and follow the instruction.',
     <br/>,
     <a href="http://www.tectono-business.com/2016/01/become-tectono-lady-of-week.html">http://www.tectono-business.com/2016/01/become-tectono-lady-of-week.html</a>]




```python
re.search('@.*',res[6])[0]
```




    '@Iceprincezamani'




```python
celeb=[]
for i in res:
    search1=re.search('@.*',str(i))
    if search1 !=None:
        celeb.append(search1[0])
```


```python
del celeb[-1]
celeb.append('@RealStaceyBanks')
celeb
```




    ['@Wizkidayo',
     '@Iceprincezamani',
     '@BankyW',
     '@Cossydiva',
     '@TONTOLET',
     '@olamide_YBNL',
     '@DONJAZZY',
     '@iamdbanj',
     '@TiwaSavage',
     '@Realomosexy',
     '@ritaUdominic',
     '@uchejombo',
     '@iam_Davido',
     '@phynofino',
     '@wandecoal',
     '@RealStaceyBanks']




```python
consumer_key = '##############'
consumer_secret = '##############'
access_token = '##############'
access_token_secret = '##############'
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

```


```python

```


```python
new_user=[]
for i in celeb:
    try:
        for user in tweepy.Cursor(api.friends, screen_name=i,wait_on_rate_limit=True).items():
                if 'Nigeria' in user.location:
                    new_user.append(user.screen_name)
        #time.sleep(60)
    except tweepy.TweepError as ex:
        if ex.reason == "Not authorized" :
            continue
```


```python
a_user=[]
for i in journalist:
    try:
        for user in tweepy.Cursor(api.friends, screen_name=i,wait_on_rate_limit=True).items():
                #print (user)
                if (user.followers_count)>200:
                    a_user.append(user.screen_name)
                #time.sleep(60)
    except tweepy.TweepError as ex:
        if ex.reason == "Not authorized" :
            continue
```


```python
b_user=[]
for i in govt_official:
    #try:
        for user in tweepy.Cursor(api.friends, screen_name=i,wait_on_rate_limit=True).items():
                print (user)
                if (user.followers_count)>200:
                    b_user.append(user.screen_name)
        #time.sleep(60)
    #except tweepy.TweepError as ex:
     #   if ex.reason == "Not authorized" :
      #      continue
```


```python
final_list=a_user+b_user+celeb+govt_official+journalist

```


```python
class tweetsearch():
    '''
    This is a basic class to search and download twitter data.
    You can build up on it to extend the functionalities for more 
    sophisticated analysis
    '''
    def __init__(self, cols=None,auth=None):
        #
        if not cols is None:
            self.cols = cols
        else:
            self.cols = ['id', 'created_at', 'source', 'original_text','clean_text', 
                    'sentiment','polarity','subjectivity', 'lang',
                    'favorite_count', 'retweet_count', 'original_author',   
                    'possibly_sensitive', 'hashtags',
                    'user_mentions', 'place', 'place_coord_boundaries']
            
        if auth is None:
            consumer_key = 'haAX9mGU6dfO5l9WXhilBNT1z'
            consumer_secret = 'tBCRdcAcjAnyZizPRqAlb2R38bDOEia3W4sOsMxrty9FRJnF8j'
            access_token = '129153692-WlHYMMwfRCsLjrshrLjcZPE0scqDtlycks9o6whO'
            access_token_secret = 'omoGw2DW7yItrnUrVT4pKfnK6vddSBuJzkYo8qHHmy5LL'
            auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
            auth.set_access_token(access_token, access_token_secret)
            
        


            #This handles Twitter authetification and the connection to Twitter Streaming API
            
            

        #            
        self.auth = auth
        self.api = tweepy.API(auth) 
        self.filtered_tweet = ''
            

    def clean_tweets(self, twitter_text):

        #use pre processor
        tweet = p.clean(twitter_text)

         #HappyEmoticons
        emoticons_happy = set([
            ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
            ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
            '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
            'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
            '<3'
            ])

        # Sad Emoticons
        emoticons_sad = set([
            ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
            ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
            ':c', ':{', '>:\\', ';('
            ])

        #Emoji patterns
        emoji_pattern = re.compile("["
                 u"\U0001F600-\U0001F64F"  # emoticons
                 u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                 u"\U0001F680-\U0001F6FF"  # transport & map symbols
                 u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                 u"\U00002702-\U000027B0"
                 u"\U000024C2-\U0001F251"
                 "]+", flags=re.UNICODE)

        #combine sad and happy emoticons
        emoticons = emoticons_happy.union(emoticons_sad)

        stop_words = set(stopwords.words('english'))
        word_tokens = nltk.word_tokenize(tweet)
        #after tweepy preprocessing the colon symbol left remain after      
        #removing mentions
        tweet = re.sub(r':', '', tweet)
        tweet = re.sub(r'‚Ä¶', '', tweet)

        #replace consecutive non-ASCII characters with a space
        tweet = re.sub(r'[^\x00-\x7F]+',' ', tweet)

        #remove emojis from tweet
        tweet = emoji_pattern.sub(r'', tweet)

        #filter using NLTK library append it to a string
        filtered_tweet = [w for w in word_tokens if not w in stop_words]

        #looping through conditions
        filtered_tweet = []    
        for w in word_tokens:
        #check tokens against stop words , emoticons and punctuations
            if w not in stop_words and w not in emoticons and w not in string.punctuation:
                filtered_tweet.append(w)

        return ' '.join(filtered_tweet)            
    def find_friends(self,i):
        test=set()
        try:
            for user in tweepy.Cursor(api.friends, screen_name=i,wait_on_rate_limit=True,geocode='6.465422,3.406448,1km').items():
                #print (user)
                if (user.statuses_count)>500:
                    test.add(user.screen_name)
        except tweepy.TweepError as e:
            pass
            print (e)
        return test
                    
    def get_tweets(self, keyword, csvfile=None):
        
        
        df = pd.DataFrame(columns=self.cols)
        
        if not csvfile is None:
            #If the file exists, then read the existing data from the CSV file.
            if os.path.exists(csvfile):
                df = pd.read_csv(csvfile, header=0)
            

        #page attribute in tweepy.cursor and iteration
        for status in tweepy.Cursor(self.api.user_timeline, screen_name=keyword, include_rts=False,wait_on_rate_limit=True,tweet_mode='extended').items(505):

            # the you receive from the Twitter API is in a JSON format and has quite an amount of information attached
            #for status in page:
                
                new_entry = []
                status = status._json
                
                #filter by language
                if status['lang'] != 'en':
                    continue

                
                #if this tweet is a retweet update retweet count
                if status['created_at'] in df['created_at'].values:
                    i = df.loc[df['created_at'] == status['created_at']].index[0]
                    #
                    cond1 = status['favorite_count'] != df.at[i, 'favorite_count']
                    cond2 = status['retweet_count'] != df.at[i, 'retweet_count']
                    if cond1 or cond2:
                        df.at[i, 'favorite_count'] = status['favorite_count']
                        df.at[i, 'retweet_count'] = status['retweet_count']
                    continue

                #calculate sentiment
                filtered_tweet = self.clean_tweets(status['full_text'])
                blob = TextBlob(filtered_tweet)
                Sentiment = blob.sentiment     
                polarity = Sentiment.polarity
                subjectivity = Sentiment.subjectivity

                new_entry += [status['id'], status['created_at'],
                              status['source'], status['full_text'], filtered_tweet, 
                              Sentiment,polarity,subjectivity, status['lang'],
                              status['favorite_count'], status['retweet_count']]

                new_entry.append(status['user']['screen_name'])

                try:
                    is_sensitive = status['possibly_sensitive']
                except KeyError:
                    is_sensitive = None

                new_entry.append(is_sensitive)

                hashtags = ", ".join([hashtag_item['text'] for hashtag_item in status['entities']['hashtags']])
                new_entry.append(hashtags) #append the hashtags

                #
                mentions = ", ".join([mention['screen_name'] for mention in status['entities']['user_mentions']])
                new_entry.append(mentions) #append the user mentions

                try:
                    xyz = status['place']['bounding_box']['coordinates']
                    coordinates = [coord for loc in xyz for coord in loc]
                except TypeError:
                    coordinates = None
                #
                new_entry.append(coordinates)

                try:
                    location = status['user']['location']
                except TypeError:
                    location = ''
                #
                new_entry.append(location)

                #now append a row to the dataframe
                single_tweet_df = pd.DataFrame([new_entry], columns=self.cols)
                df = df.append(single_tweet_df, ignore_index=True)

        if not csvfile is None:
            #save it to file
            df.to_csv(csvfile, columns=self.cols, index=False, encoding="utf-8")
            
        return df
```


```python
#final_list=set(list(test)+final_list)

```


```python
ts=tweetsearch()
```


```python
test=[]
for i in final_list:
    if (len(final_list)+len(final_list))<1000:
        test.extend(ts.find_friends(i))
final_list.extend(test)
```


```python
for i in final_list:
    try:
        a=ts.get_tweets(i)
        a['user']=i
        I_A=pd.concat([a,I_A])
    except NameError:
        I_A=a
    except tweepy.TweepError as e:
        print (e)
        pass
```

    Failed to send request: ('Connection aborted.', ConnectionResetError(10054, 'An existing connection was forcibly closed by the remote host', None, 10054, None))
    Failed to send request: ('Connection aborted.', ConnectionResetError(10054, 'An existing connection was forcibly closed by the remote host', None, 10054, None))
    Failed to send request: ('Connection aborted.', ConnectionResetError(10054, 'An existing connection was forcibly closed by the remote host', None, 10054, None))
    Failed to send request: ('Connection aborted.', ConnectionResetError(10054, 'An existing connection was forcibly closed by the remote host', None, 10054, None))
    Failed to send request: ('Connection aborted.', ConnectionResetError(10054, 'An existing connection was forcibly closed by the remote host', None, 10054, None))
    Failed to send request: ('Connection aborted.', ConnectionResetError(10054, 'An existing connection was forcibly closed by the remote host', None, 10054, None))
    Failed to send request: ('Connection aborted.', ConnectionResetError(10054, 'An existing connection was forcibly closed by the remote host', None, 10054, None))
    Failed to send request: HTTPSConnectionPool(host='api.twitter.com', port=443): Max retries exceeded with url: /1.1/statuses/user_timeline.json?max_id=1146073033213390847&screen_name=theresa_may&include_rts=False&tweet_mode=extended (Caused by NewConnectionError('<urllib3.connection.VerifiedHTTPSConnection object at 0x000002042AE626A0>: Failed to establish a new connection: [Errno 11001] getaddrinfo failed'))
    Failed to send request: HTTPSConnectionPool(host='api.twitter.com', port=443): Max retries exceeded with url: /1.1/statuses/user_timeline.json?screen_name=CARE&include_rts=False&tweet_mode=extended (Caused by NewConnectionError('<urllib3.connection.VerifiedHTTPSConnection object at 0x000002042B0487B8>: Failed to establish a new connection: [Errno 11001] getaddrinfo failed'))
    Failed to send request: HTTPSConnectionPool(host='api.twitter.com', port=443): Max retries exceeded with url: /1.1/statuses/user_timeline.json?screen_name=AzebWG&include_rts=False&tweet_mode=extended (Caused by NewConnectionError('<urllib3.connection.VerifiedHTTPSConnection object at 0x000002042B0485F8>: Failed to establish a new connection: [Errno 11001] getaddrinfo failed'))
    Failed to send request: HTTPSConnectionPool(host='api.twitter.com', port=443): Max retries exceeded with url: /1.1/statuses/user_timeline.json?screen_name=PaulChapman_&include_rts=False&tweet_mode=extended (Caused by NewConnectionError('<urllib3.connection.VerifiedHTTPSConnection object at 0x000002042B048D68>: Failed to establish a new connection: [Errno 11001] getaddrinfo failed'))
    Failed to send request: HTTPSConnectionPool(host='api.twitter.com', port=443): Max retries exceeded with url: /1.1/statuses/user_timeline.json?screen_name=NicolaCareem&include_rts=False&tweet_mode=extended (Caused by NewConnectionError('<urllib3.connection.VerifiedHTTPSConnection object at 0x000002042A9FB550>: Failed to establish a new connection: [Errno 11001] getaddrinfo failed'))
    Failed to send request: HTTPSConnectionPool(host='api.twitter.com', port=443): Max retries exceeded with url: /1.1/statuses/user_timeline.json?screen_name=orlandopirates&include_rts=False&tweet_mode=extended (Caused by NewConnectionError('<urllib3.connection.VerifiedHTTPSConnection object at 0x000002042B092B00>: Failed to establish a new connection: [Errno 11001] getaddrinfo failed'))
    Failed to send request: HTTPSConnectionPool(host='api.twitter.com', port=443): Max retries exceeded with url: /1.1/statuses/user_timeline.json?screen_name=EFA&include_rts=False&tweet_mode=extended (Caused by NewConnectionError('<urllib3.connection.VerifiedHTTPSConnection object at 0x000002042AC1A208>: Failed to establish a new connection: [Errno 11001] getaddrinfo failed'))
    Failed to send request: HTTPSConnectionPool(host='api.twitter.com', port=443): Max retries exceeded with url: /1.1/statuses/user_timeline.json?screen_name=StevenLevy&include_rts=False&tweet_mode=extended (Caused by NewConnectionError('<urllib3.connection.VerifiedHTTPSConnection object at 0x000002042AC1A6D8>: Failed to establish a new connection: [Errno 11001] getaddrinfo failed'))
    Failed to send request: HTTPSConnectionPool(host='api.twitter.com', port=443): Max retries exceeded with url: /1.1/statuses/user_timeline.json?screen_name=Argentina&include_rts=False&tweet_mode=extended (Caused by NewConnectionError('<urllib3.connection.VerifiedHTTPSConnection object at 0x000002042B06D358>: Failed to establish a new connection: [Errno 11001] getaddrinfo failed'))
    Failed to send request: HTTPSConnectionPool(host='api.twitter.com', port=443): Max retries exceeded with url: /1.1/statuses/user_timeline.json?screen_name=gdarmsta&include_rts=False&tweet_mode=extended (Caused by NewConnectionError('<urllib3.connection.VerifiedHTTPSConnection object at 0x000002042B092278>: Failed to establish a new connection: [Errno 11001] getaddrinfo failed'))
    Failed to send request: HTTPSConnectionPool(host='api.twitter.com', port=443): Max retries exceeded with url: /1.1/statuses/user_timeline.json?screen_name=Mortada5Mansour&include_rts=False&tweet_mode=extended (Caused by NewConnectionError('<urllib3.connection.VerifiedHTTPSConnection object at 0x000002042B048128>: Failed to establish a new connection: [Errno 11001] getaddrinfo failed'))
    Failed to send request: HTTPSConnectionPool(host='api.twitter.com', port=443): Max retries exceeded with url: /1.1/statuses/user_timeline.json?screen_name=austintylerro&include_rts=False&tweet_mode=extended (Caused by NewConnectionError('<urllib3.connection.VerifiedHTTPSConnection object at 0x000002042B048748>: Failed to establish a new connection: [Errno 11001] getaddrinfo failed'))
    Failed to send request: HTTPSConnectionPool(host='api.twitter.com', port=443): Max retries exceeded with url: /1.1/statuses/user_timeline.json?screen_name=HeyBuckHey&include_rts=False&tweet_mode=extended (Caused by NewConnectionError('<urllib3.connection.VerifiedHTTPSConnection object at 0x000002042B048AC8>: Failed to establish a new connection: [Errno 11001] getaddrinfo failed'))
    Failed to send request: HTTPSConnectionPool(host='api.twitter.com', port=443): Max retries exceeded with url: /1.1/statuses/user_timeline.json?screen_name=kaikahele&include_rts=False&tweet_mode=extended (Caused by NewConnectionError('<urllib3.connection.VerifiedHTTPSConnection object at 0x000002042AE24E80>: Failed to establish a new connection: [Errno 11001] getaddrinfo failed'))
    Failed to send request: HTTPSConnectionPool(host='api.twitter.com', port=443): Max retries exceeded with url: /1.1/statuses/user_timeline.json?screen_name=EricHorngABC7&include_rts=False&tweet_mode=extended (Caused by NewConnectionError('<urllib3.connection.VerifiedHTTPSConnection object at 0x000002042B02DE10>: Failed to establish a new connection: [Errno 11001] getaddrinfo failed'))
    Failed to send request: HTTPSConnectionPool(host='api.twitter.com', port=443): Max retries exceeded with url: /1.1/statuses/user_timeline.json?screen_name=harari_yuval&include_rts=False&tweet_mode=extended (Caused by NewConnectionError('<urllib3.connection.VerifiedHTTPSConnection object at 0x000002042AB7F160>: Failed to establish a new connection: [Errno 11001] getaddrinfo failed'))
    Failed to send request: HTTPSConnectionPool(host='api.twitter.com', port=443): Max retries exceeded with url: /1.1/statuses/user_timeline.json?screen_name=wabote_simbi&include_rts=False&tweet_mode=extended (Caused by NewConnectionError('<urllib3.connection.VerifiedHTTPSConnection object at 0x000002042AEA1358>: Failed to establish a new connection: [Errno 11001] getaddrinfo failed'))
    Failed to send request: HTTPSConnectionPool(host='api.twitter.com', port=443): Max retries exceeded with url: /1.1/statuses/user_timeline.json?screen_name=IAVI&include_rts=False&tweet_mode=extended (Caused by NewConnectionError('<urllib3.connection.VerifiedHTTPSConnection object at 0x000002042B048860>: Failed to establish a new connection: [Errno 11001] getaddrinfo failed'))
    Failed to send request: HTTPSConnectionPool(host='api.twitter.com', port=443): Max retries exceeded with url: /1.1/statuses/user_timeline.json?screen_name=GlblCtznImpact&include_rts=False&tweet_mode=extended (Caused by NewConnectionError('<urllib3.connection.VerifiedHTTPSConnection object at 0x000002042A84F5F8>: Failed to establish a new connection: [Errno 11001] getaddrinfo failed'))
    Failed to send request: HTTPSConnectionPool(host='api.twitter.com', port=443): Max retries exceeded with url: /1.1/statuses/user_timeline.json?screen_name=NickKristof&include_rts=False&tweet_mode=extended (Caused by NewConnectionError('<urllib3.connection.VerifiedHTTPSConnection object at 0x000002042A79A080>: Failed to establish a new connection: [Errno 11001] getaddrinfo failed'))
    Failed to send request: HTTPSConnectionPool(host='api.twitter.com', port=443): Max retries exceeded with url: /1.1/statuses/user_timeline.json?screen_name=SueDHellmann&include_rts=False&tweet_mode=extended (Caused by NewConnectionError('<urllib3.connection.VerifiedHTTPSConnection object at 0x000002042B0ACD30>: Failed to establish a new connection: [Errno 11001] getaddrinfo failed'))
    Failed to send request: HTTPSConnectionPool(host='api.twitter.com', port=443): Max retries exceeded with url: /1.1/statuses/user_timeline.json?screen_name=GatesAfrica&include_rts=False&tweet_mode=extended (Caused by NewConnectionError('<urllib3.connection.VerifiedHTTPSConnection object at 0x000002042AE242E8>: Failed to establish a new connection: [Errno 11001] getaddrinfo failed'))
    Failed to send request: HTTPSConnectionPool(host='api.twitter.com', port=443): Max retries exceeded with url: /1.1/statuses/user_timeline.json?screen_name=EgyptianPlayers&include_rts=False&tweet_mode=extended (Caused by NewConnectionError('<urllib3.connection.VerifiedHTTPSConnection object at 0x000002042AE244E0>: Failed to establish a new connection: [Errno 11001] getaddrinfo failed'))
    Failed to send request: HTTPSConnectionPool(host='api.twitter.com', port=443): Max retries exceeded with url: /1.1/statuses/user_timeline.json?screen_name=NHGSFP&include_rts=False&tweet_mode=extended (Caused by NewConnectionError('<urllib3.connection.VerifiedHTTPSConnection object at 0x000002042B048438>: Failed to establish a new connection: [Errno 11001] getaddrinfo failed'))
    Failed to send request: HTTPSConnectionPool(host='api.twitter.com', port=443): Max retries exceeded with url: /1.1/statuses/user_timeline.json?screen_name=ESTuniscom&include_rts=False&tweet_mode=extended (Caused by NewConnectionError('<urllib3.connection.VerifiedHTTPSConnection object at 0x000002042B0480B8>: Failed to establish a new connection: [Errno 11001] getaddrinfo failed'))
    Failed to send request: HTTPSConnectionPool(host='api.twitter.com', port=443): Max retries exceeded with url: /1.1/statuses/user_timeline.json?screen_name=MagassoubaMakan&include_rts=False&tweet_mode=extended (Caused by NewConnectionError('<urllib3.connection.VerifiedHTTPSConnection object at 0x000002042ACDBCC0>: Failed to establish a new connection: [Errno 11001] getaddrinfo failed'))
    Failed to send request: HTTPSConnectionPool(host='api.twitter.com', port=443): Max retries exceeded with url: /1.1/statuses/user_timeline.json?screen_name=Airbnb&include_rts=False&tweet_mode=extended (Caused by NewConnectionError('<urllib3.connection.VerifiedHTTPSConnection object at 0x000002042B092EF0>: Failed to establish a new connection: [Errno 11001] getaddrinfo failed'))
    Failed to send request: HTTPSConnectionPool(host='api.twitter.com', port=443): Max retries exceeded with url: /1.1/statuses/user_timeline.json?screen_name=PaulineKTallen&include_rts=False&tweet_mode=extended (Caused by NewConnectionError('<urllib3.connection.VerifiedHTTPSConnection object at 0x000002042AC1A550>: Failed to establish a new connection: [Errno 11001] getaddrinfo failed'))
    Failed to send request: HTTPSConnectionPool(host='api.twitter.com', port=443): Max retries exceeded with url: /1.1/statuses/user_timeline.json?screen_name=Q13FOXKiggins&include_rts=False&tweet_mode=extended (Caused by NewConnectionError('<urllib3.connection.VerifiedHTTPSConnection object at 0x000002042B23B198>: Failed to establish a new connection: [Errno 11001] getaddrinfo failed'))
    Failed to send request: HTTPSConnectionPool(host='api.twitter.com', port=443): Max retries exceeded with url: /1.1/statuses/user_timeline.json?screen_name=NunesAlt&include_rts=False&tweet_mode=extended (Caused by NewConnectionError('<urllib3.connection.VerifiedHTTPSConnection object at 0x000002042B06D2E8>: Failed to establish a new connection: [Errno 11001] getaddrinfo failed'))
    Failed to send request: HTTPSConnectionPool(host='api.twitter.com', port=443): Max retries exceeded with url: /1.1/statuses/user_timeline.json?screen_name=HYPREPNigeria&include_rts=False&tweet_mode=extended (Caused by NewConnectionError('<urllib3.connection.VerifiedHTTPSConnection object at 0x000002042B0480B8>: Failed to establish a new connection: [Errno 11001] getaddrinfo failed'))
    Failed to send request: HTTPSConnectionPool(host='api.twitter.com', port=443): Max retries exceeded with url: /1.1/statuses/user_timeline.json?screen_name=governorobaseki&include_rts=False&tweet_mode=extended (Caused by NewConnectionError('<urllib3.connection.VerifiedHTTPSConnection object at 0x000002042B048A58>: Failed to establish a new connection: [Errno 11001] getaddrinfo failed'))
    Failed to send request: HTTPSConnectionPool(host='api.twitter.com', port=443): Max retries exceeded with url: /1.1/statuses/user_timeline.json?screen_name=JosephSteinberg&include_rts=False&tweet_mode=extended (Caused by NewConnectionError('<urllib3.connection.VerifiedHTTPSConnection object at 0x000002042AE247F0>: Failed to establish a new connection: [Errno 11001] getaddrinfo failed'))
    Failed to send request: HTTPSConnectionPool(host='api.twitter.com', port=443): Max retries exceeded with url: /1.1/statuses/user_timeline.json?screen_name=wsary&include_rts=False&tweet_mode=extended (Caused by NewConnectionError('<urllib3.connection.VerifiedHTTPSConnection object at 0x000002042AE242E8>: Failed to establish a new connection: [Errno 11001] getaddrinfo failed'))
    Failed to send request: HTTPSConnectionPool(host='api.twitter.com', port=443): Max retries exceeded with url: /1.1/statuses/user_timeline.json?screen_name=yangconomics&include_rts=False&tweet_mode=extended (Caused by NewConnectionError('<urllib3.connection.VerifiedHTTPSConnection object at 0x000002042B02DE10>: Failed to establish a new connection: [Errno 11001] getaddrinfo failed'))
    Failed to send request: HTTPSConnectionPool(host='api.twitter.com', port=443): Max retries exceeded with url: /1.1/statuses/user_timeline.json?screen_name=%40iam_Davido&include_rts=False&tweet_mode=extended (Caused by NewConnectionError('<urllib3.connection.VerifiedHTTPSConnection object at 0x000002042AEA1358>: Failed to establish a new connection: [Errno 11001] getaddrinfo failed'))
    Failed to send request: HTTPSConnectionPool(host='api.twitter.com', port=443): Max retries exceeded with url: /1.1/statuses/user_timeline.json?screen_name=emesola&include_rts=False&tweet_mode=extended (Caused by NewConnectionError('<urllib3.connection.VerifiedHTTPSConnection object at 0x000002042B095048>: Failed to establish a new connection: [Errno 11001] getaddrinfo failed'))
    Failed to send request: HTTPSConnectionPool(host='api.twitter.com', port=443): Max retries exceeded with url: /1.1/statuses/user_timeline.json?screen_name=IbeKachikwu&include_rts=False&tweet_mode=extended (Caused by NewConnectionError('<urllib3.connection.VerifiedHTTPSConnection object at 0x000002042AE24D68>: Failed to establish a new connection: [Errno 11001] getaddrinfo failed'))
    Failed to send request: HTTPSConnectionPool(host='api.twitter.com', port=443): Max retries exceeded with url: /1.1/statuses/user_timeline.json?screen_name=ElizKolbert&include_rts=False&tweet_mode=extended (Caused by NewConnectionError('<urllib3.connection.VerifiedHTTPSConnection object at 0x000002042AEA1A20>: Failed to establish a new connection: [Errno 11001] getaddrinfo failed'))
    Failed to send request: HTTPSConnectionPool(host='api.twitter.com', port=443): Max retries exceeded with url: /1.1/statuses/user_timeline.json?screen_name=TheAEIC&include_rts=False&tweet_mode=extended (Caused by NewConnectionError('<urllib3.connection.VerifiedHTTPSConnection object at 0x000002042B06D358>: Failed to establish a new connection: [Errno 11001] getaddrinfo failed'))
    Failed to send request: HTTPSConnectionPool(host='api.twitter.com', port=443): Max retries exceeded with url: /1.1/statuses/user_timeline.json?screen_name=KateWhineHall&include_rts=False&tweet_mode=extended (Caused by NewConnectionError('<urllib3.connection.VerifiedHTTPSConnection object at 0x000002042AAC4128>: Failed to establish a new connection: [Errno 11001] getaddrinfo failed'))
    Failed to send request: HTTPSConnectionPool(host='api.twitter.com', port=443): Max retries exceeded with url: /1.1/statuses/user_timeline.json?screen_name=UrugwiroVillage&include_rts=False&tweet_mode=extended (Caused by NewConnectionError('<urllib3.connection.VerifiedHTTPSConnection object at 0x000002042B02DEB8>: Failed to establish a new connection: [Errno 11001] getaddrinfo failed'))
    Failed to send request: ('Connection aborted.', ConnectionResetError(10054, 'An existing connection was forcibly closed by the remote host', None, 10054, None))
    Failed to send request: ('Connection aborted.', ConnectionResetError(10054, 'An existing connection was forcibly closed by the remote host', None, 10054, None))
    Failed to send request: ('Connection aborted.', ConnectionResetError(10054, 'An existing connection was forcibly closed by the remote host', None, 10054, None))
    Failed to send request: ('Connection aborted.', ConnectionResetError(10054, 'An existing connection was forcibly closed by the remote host', None, 10054, None))
    Failed to send request: ('Connection aborted.', ConnectionResetError(10054, 'An existing connection was forcibly closed by the remote host', None, 10054, None))
    Failed to send request: ('Connection aborted.', ConnectionResetError(10054, 'An existing connection was forcibly closed by the remote host', None, 10054, None))
    Failed to send request: ('Connection aborted.', ConnectionResetError(10054, 'An existing connection was forcibly closed by the remote host', None, 10054, None))
    Failed to send request: ('Connection aborted.', ConnectionResetError(10054, 'An existing connection was forcibly closed by the remote host', None, 10054, None))
    


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-26-03483de1d09b> in <module>
          1 for i in final_list:
          2     try:
    ----> 3         a=ts.get_tweets(i)
          4         a['user']=i
          5         I_A=pd.concat([a,I_A])
    

    <ipython-input-17-fd7f428827c0> in get_tweets(self, keyword, csvfile)
        119 
        120         #page attribute in tweepy.cursor and iteration
    --> 121         for status in tweepy.Cursor(self.api.user_timeline, screen_name=keyword, include_rts=False,wait_on_rate_limit=True,tweet_mode='extended').items(505):
        122 
        123             # the you receive from the Twitter API is in a JSON format and has quite an amount of information attached
    

    ~\Anaconda3\lib\site-packages\tweepy\cursor.py in __next__(self)
         45 
         46     def __next__(self):
    ---> 47         return self.next()
         48 
         49     def next(self):
    

    ~\Anaconda3\lib\site-packages\tweepy\cursor.py in next(self)
        193         if self.current_page is None or self.page_index == len(self.current_page) - 1:
        194             # Reached end of current page, get the next page...
    --> 195             self.current_page = self.page_iterator.next()
        196             self.page_index = -1
        197         self.page_index += 1
    

    ~\Anaconda3\lib\site-packages\tweepy\cursor.py in next(self)
        104 
        105         if self.index >= len(self.results) - 1:
    --> 106             data = self.method(max_id=self.max_id, parser=RawParser(), *self.args, **self.kargs)
        107 
        108             if hasattr(self.method, '__self__'):
    

    ~\Anaconda3\lib\site-packages\tweepy\binder.py in _call(*args, **kwargs)
        248                 return method
        249             else:
    --> 250                 return method.execute()
        251         finally:
        252             method.session.close()
    

    ~\Anaconda3\lib\site-packages\tweepy\binder.py in execute(self)
        187                                                 timeout=self.api.timeout,
        188                                                 auth=auth,
    --> 189                                                 proxies=self.api.proxy)
        190                 except Exception as e:
        191                     six.reraise(TweepError, TweepError('Failed to send request: %s' % e), sys.exc_info()[2])
    

    ~\Anaconda3\lib\site-packages\requests\sessions.py in request(self, method, url, params, data, headers, cookies, files, auth, timeout, allow_redirects, proxies, hooks, stream, verify, cert, json)
        528         }
        529         send_kwargs.update(settings)
    --> 530         resp = self.send(prep, **send_kwargs)
        531 
        532         return resp
    

    ~\Anaconda3\lib\site-packages\requests\sessions.py in send(self, request, **kwargs)
        683 
        684         if not stream:
    --> 685             r.content
        686 
        687         return r
    

    ~\Anaconda3\lib\site-packages\requests\models.py in content(self)
        827                 self._content = None
        828             else:
    --> 829                 self._content = b''.join(self.iter_content(CONTENT_CHUNK_SIZE)) or b''
        830 
        831         self._content_consumed = True
    

    ~\Anaconda3\lib\site-packages\requests\models.py in generate()
        749             if hasattr(self.raw, 'stream'):
        750                 try:
    --> 751                     for chunk in self.raw.stream(chunk_size, decode_content=True):
        752                         yield chunk
        753                 except ProtocolError as e:
    

    ~\Anaconda3\lib\site-packages\urllib3\response.py in stream(self, amt, decode_content)
        492         else:
        493             while not is_fp_closed(self._fp):
    --> 494                 data = self.read(amt=amt, decode_content=decode_content)
        495 
        496                 if data:
    

    ~\Anaconda3\lib\site-packages\urllib3\response.py in read(self, amt, decode_content, cache_content)
        440             else:
        441                 cache_content = False
    --> 442                 data = self._fp.read(amt)
        443                 if amt != 0 and not data:  # Platform-specific: Buggy versions of Python.
        444                     # Close the connection when no data is returned
    

    ~\Anaconda3\lib\http\client.py in read(self, amt)
        445             # Amount is given, implement using readinto
        446             b = bytearray(amt)
    --> 447             n = self.readinto(b)
        448             return memoryview(b)[:n].tobytes()
        449         else:
    

    ~\Anaconda3\lib\http\client.py in readinto(self, b)
        489         # connection, and the user is reading more bytes than will be provided
        490         # (for example, reading in 1k chunks)
    --> 491         n = self.fp.readinto(b)
        492         if not n and b:
        493             # Ideally, we would raise IncompleteRead if the content-length
    

    ~\Anaconda3\lib\socket.py in readinto(self, b)
        587         while True:
        588             try:
    --> 589                 return self._sock.recv_into(b)
        590             except timeout:
        591                 self._timeout_occurred = True
    

    ~\Anaconda3\lib\ssl.py in recv_into(self, buffer, nbytes, flags)
       1050                   "non-zero flags not allowed in calls to recv_into() on %s" %
       1051                   self.__class__)
    -> 1052             return self.read(nbytes, buffer)
       1053         else:
       1054             return super().recv_into(buffer, nbytes, flags)
    

    ~\Anaconda3\lib\ssl.py in read(self, len, buffer)
        909         try:
        910             if buffer is not None:
    --> 911                 return self._sslobj.read(len, buffer)
        912             else:
        913                 return self._sslobj.read(len)
    

    KeyboardInterrupt: 



```python
#final_list=pd.DataFrame({'final_list':final_list})
#final_list.to_csv('final.csv')
#final_list=pd.read_csv('./final.csv')
#final_list=list(final_list.final_list)

```


```python
df=I_A
len(df.user.unique())
```

    C:\Users\Hp\Anaconda3\lib\site-packages\IPython\core\interactiveshell.py:3057: DtypeWarning: Columns (1,8,10,11,19) have mixed types. Specify dtype option on import or set low_memory=False.
      interactivity=interactivity, compiler=compiler, result=result)
    




    959




```python
data = df.original_text.values.tolist()
data = [re.sub('\S*@\S*\s?', '', str(sent)) for sent in data]
data = [re.sub("\'", "", str(sent)) for sent in data]
data

```




    ['Truly is',
     'My information comes from helping physicians fill out death certificates for the last 25 years.',
     'Yes my Dad taught me',
     'I believe my brain is loud enough already, no thanks',
     'Yes precisely I just zoomed in on the original',
     'It was on one of the officers uniforms in Portland',
     'Can I just say, I need a hug, Im sure we all do right now',
     'Obama always held the umbrella for Michelle and other women when ever I saw it raining and he was walking https://t.co/B0zRU8vx84',
     'My daughters took 13 days in L.A. too and that was in the beginning of July',
     'Thank you',
     'I love that open opening statement',
     'ð\x9f¤® My husband and I wanted one but not anymore',
     'They are not rioting, just STOP',
     'Do you think she gets the oxymoron?',
     'There are way too many to list in one tweet, anything specific in mind? Perhaps then I MAY be able to fit it into one tweet',
     'What are you talking about? I grew up in the inner city, what does that have to do with affecting change? No this is not a repeat of history, I grew up 4 blocks from where the LA riots started in 92. That was a R',
     'Which means in reality it is an inflated market',
     'I have to say one of the more compelling moments for me was when Trump denies Russian election interference and Putin sort of shifts and grins like he cant hardly contain himself. Everything else is very compelling but that offers a partic',
     'Oh for sure, I had never heard of this before now though',
     'Makes sense now',
     'It doesnt count as a covid19 death only towards the infection rate ð\x9f¤¦â\x80\x8dâ\x99\x80ï¸\x8f jeez',
     'So we cant have freedom AND human rights its either or? Wow have you not read the constitution?',
     'Seriously? ð\x9f¤£ð\x9f¤£',
     'I saw this in another picture, what kind of flag is this?? https://t.co/7W5Bq0JHnS',
     'Wow, the judge will throw it out',
     'ð\x9f\x98³ probably not anymore https://t.co/j72rBwFzdp',
     'Hopefully Sams doesnt settle, most employers do',
     'This is BS the physician couldnt of done it that quickly with a police officer there, this is not TV. Are people really this gullible? It takes a minimum 2 days for the physician to sign it ð\x9f¤¦â\x80\x8dâ\x99\x80ï¸\x8f',
     'Whatever attorney took her case is dumb. The legal precedent is "what would a logical person do?" During a pandemic a logical person Wears a mask ð\x9f¤\x94ð\x9f¤¦â\x80\x8dâ\x99\x80ï¸\x8f',
     'Most do that, what I mean is that he shouldnt have released the names on the list. I think it has caused a lot of frustration rather than calming people as Im sure was the intention. To give us hope, but it feels like an eternity',
     'I dont like the way the announced his VP selection and who is there etc. I dont think it has been very productive actually the opposite',
     'This really should be trending #AOCspeaks4Me',
     'Calculated not weird it is calculated',
     'Perhaps lead with that to avoid confusion because those first couple of tweets ð\x9f¤¦â\x80\x8dâ\x99\x80ï¸\x8f',
     'Its been long past time, enough is enough',
     'Florida, do expect anything else at this point?',
     'Most of us just cringe at the memory of long division, we are just envious',
     'Im losing sleep over it',
     'ð\x9f\x98³ I havent even fell asleep yet',
     'I live in L.A. and even I know it isnt fixed.',
     'You need help, please get it',
     'You need help, please get it',
     'Now this I probably imagined happening in 2020 when I was a kid',
     'Safe place? No my Dad died in 2011, I will never be in a safe place again. Pinche safe place',
     'Honestly I just opened it and read the title and thought of course this is the title ð\x9f¤¦â\x80\x8dâ\x99\x80ï¸\x8f',
     'I get that damn lump gets me too, which in turn makes me more mad',
     'Yeah I thought that was a tad extreme even by Twitterverse standards too',
     'What do you mean?',
     'In case youre not, shame on you for judging others based upon a tweet, Im a million % sure Christ wouldnt approve',
     'ð\x9f¤¢ð\x9f¤® tequila, when did they start putting into hand sanitizer?',
     'We have to follow our moral and ethical convictions in healthcare. Speak out about your reservations, you conscious will thank you later',
     'God bless his soul! This spoke to my heart #LastBornsUnite ð\x9f\x99\x8c',
     'schools should not open until there is a national testing and tracing strategy.',
     'The founder of ByteDance, TikToks parent company, wanted to build a global software giant. Now that ambition is in jeopardy https://t.co/Ry3uClUc5T',
     'On â\x80\x9cBabbageâ\x80\x9d:\n\n-Professor Sarah Gilbert of on the timeline for an effective covid-19 vaccine\n-Navigating the sky with diamonds\n-And why sewage is helping census-takers \n\nhttps://t.co/f1rEMWfwR1',
     'Is Hong Kongâ\x80\x99s new national-security law the end of â\x80\x9cone country, two systemsâ\x80\x9d? asks pro-democracy activist and pro-Beijing Hong Kong politician Regina Ip on â\x80\x9cThe Economist Asksâ\x80\x9d https://t.co/FJcP95fT68',
     'Americas Midwest has a population equal to Britains. Its economyâ\x80\x94worth some $4trnâ\x80\x94is equal to Germanys https://t.co/ONSQmfoIlg',
     'Seafishing in Britain accounts for only 0.1% of GDP and barely 12,000 jobs. Yet it could scupper the chances of a Brexit deal https://t.co/V0J27i2ucB',
     'Tanzania forecasts GDP growth of 5.5%, making it one of the worldâ\x80\x99s star economies. But the closer you look, the less plausible these figures seem https://t.co/6n5a2wZDgR',
     'Bats are able to carry a huge diversity of viruses without getting sick, and are also more mobile than people realise https://t.co/6Yx7lb8fyF',
     'For Keir Starmer, showing that the Labour party has changed is a tricky but essential task https://t.co/ieDizJkEJu',
     'Early and extreme seasonal floods have inundated China. On â\x80\x9cThe Intelligenceâ\x80\x9d says that raises questions about the countryâ\x80\x99s river-management promises https://t.co/O8sBgMKao2 https://t.co/GAMIHMslF6',
     'Europes â\x82¬750bn covid rescue package is probably the EUâ\x80\x99s biggest constitutional leap since the creation of the euro. Could there be more? https://t.co/XggQMGShIK',
     'In the wreckage left by the coronavirus pandemic, a new era of macroeconomics is beginning. What does it hold? https://t.co/gQBISKckO6',
     'The world has entered an era of low oil pricesâ\x80\x94and no region will be more affected than the Middle East and north Africa https://t.co/T3APrf01uk',
     'The deal ultimately happened because Emmanuel Macron and Angela Merkel managedâ\x80\x94in a crisisâ\x80\x94to settle their differences beforehand https://t.co/e0EMT1eIyN',
     'Dozens of ghost sightings have been reported at the British Museum in recent years https://t.co/BPhnlfAcU6 ',
     'Prosecutors in Switzerland and Spain are investigating suspicious payments connected to the bank accounts of the former king https://t.co/aTzXqRCtvg',
     'Life on Mars could be the most significant discovery in the history of biology. Contamination from Earth puts that scientific bounty at risk https://t.co/DDl3S1b5dV',
     'Previous attempts to study organic molecules on Mars have been plagued by the presence of chemicals called perchlorates. Researchers seem to have found a solution https://t.co/EdG721C57U',
     'More than 60 big miscarriages of justice in China have been made public since Xi Jinping took power in 2012 https://t.co/p43bj5zqgt',
     'New data suggest that it takes more than a global pandemic and the biggest collapse in GDP since the 18th century to slow the British housing market https://t.co/XmLU1ybqDu',
     'Pandemic-driven insomnia has led to a boom in bedtime stories...for adults. tells â\x80\x9cThe Intelligenceâ\x80\x9d she isnâ\x80\x99t too fond of the one read by John McEnroe https://t.co/c7lpkv7AB8 https://t.co/tveRjgLwli',
     'Many Britons are enjoying being on furloughâ\x80\x94but it has some less-discussed drawbacks https://t.co/sU0xUeDwV1',
     'The IMF predicts that rich countries will borrow 17% of their combined GDP this year to fund $4.2trn in spending and tax cuts to keep the economy going https://t.co/BmbYc3fZ0b',
     'Has China won the battle for Hong Kong? On â\x80\x9cThe Economist Asksâ\x80\x9d, talks to Regina Ip, a pro-Beijing member of Hong Kongâ\x80\x99s Executive Council, and a leading pro-democracy activist https://t.co/tCUKATn5RK',
     'Russia sows discord and undermines institutions in many Western countries, but Britain is a particular target https://t.co/7uS2BnlrMk',
     'The Midwest can build on past progress if its more successful cities can reinforce what they started to get right. Read our special report on the region https://t.co/ZWUcngnuVA',
     'On â\x80\x9cThe World Aheadâ\x80\x9d podcast, the United Nationsâ\x80\x99 discusses how countries should prepare for future disasters https://t.co/3p6eNgTDZ1',
     'The programme agreed to in Brussels does more to strengthen the European Union than anyone would have imagined a few months ago  https://t.co/Gmpdc3MO3V',
     'Why a successful product looks as if its not meant to be sold https://t.co/0snT1rFCUD From ',
     'TikTok has been downloaded nearly 2bn times worldwide. Now its at risk of being banned in America. On â\x80\x9cMoney Talksâ\x80\x9d, explains what the Chinese-owned firm is doing to address concerns over privacy and security https://t.co/eO5j6DzzmP',
     'Israelâ\x80\x99s prime minister has been loudly boasting of his covid-19 successes. Now, tells â\x80\x9cThe Intelligenceâ\x80\x9d, cases are multiplyingâ\x80\x94as are protests https://t.co/YFtuC68lmb',
     'The Three Gorges Dam has failed to protect China in its struggle against devastating floods, despite suggestions that it could https://t.co/NvW63D3Flv',
     'It is not obvious that by stoking more violence in more cities Donald Trump would expand his support. Most Americans support the protests https://t.co/r2kX3oPsaE',
     'Oxford Universityâ\x80\x99s candidate vaccine shows promise in early trials. On â\x80\x9cBabbageâ\x80\x9d The Economistâ\x80\x99s health-policy editor asks lead researcher Sarah Gilbert what comes next https://t.co/fC58lhv1Cf',
     'The battle against coronavirus resembles not the front line but the Home Front https://t.co/eyFbcsPh4w From ',
     'Can planting trees offset carbon emissions produced by burning fossil fuels? https://t.co/TWD9JePt53 https://t.co/TrgVurCLeo',
     'Today on â\x80\x9cThe Intelligenceâ\x80\x9d: Israelâ\x80\x99s covid-19 spike brings protests, extreme flooding in China threatens millions and a boom in bedtime stories...for adults https://t.co/sj1Dm4aDdN',
     'America has already spent 13% of GDP on fiscal stimulus, with more on the way https://t.co/gRJk12fUBM',
     'Fewer shopping, dining and holiday options means many households in the rich world are saving money. On â\x80\x9cMoney Talksâ\x80\x9d, Britain economics correspondent says the easing of lockdowns wonâ\x80\x99t result in a rush to splash the cash https://t.co/XCcbr16Zgl https://t.co/3rUKYBEBJo',
     'You used to be able to walk into a cafÃ© and get a coffee with minimal fuss. These days ordering a drink requires knowledge of a sophisticated lexicon https://t.co/Xii01Zt3Aa From ',
     'Early and extreme seasonal floods have inundated China. On â\x80\x9cThe Intelligenceâ\x80\x9d says that raises questions about the countryâ\x80\x99s river-management promises https://t.co/1ZEfGNqSjy https://t.co/Iop7PIFN0o',
     'A profound shift is taking place in economicsâ\x80\x94the sort that happens only once in a generation https://t.co/V9MdiPFiaQ',
     'Oxford Universitys is ahead in the race to develop a covid-19 vaccine, but researcher Sarah Gilbert says changing virus transmission is delaying some trial results. Hear more on â\x80\x9cBabbageâ\x80\x9d https://t.co/bVhqNNxRwz',
     'Shanghaiâ\x80\x99s STAR market, an exchange for Chinese tech stocks, ranks second globally by capital raised in IPOs so far this year. It is just one year old https://t.co/A4cyyX8D1l',
     'â\x80\x9cBack in Blackâ\x80\x9d, which was released 40 years ago, made AC/DC international stars. It has sold 29.4m copies https://t.co/knDmh7OquY',
     'Could a covid-19 vaccine change the course of Americas presidential election? Sign up to Checks and Balance, our weekly email on US politics, for fair-minded analysis of the campaign https://t.co/la41rRcxvj',
     'The coronavirus pandemic has forced many people to make difficult judgments. On our latest â\x80\x9cMoney Talksâ\x80\x9d podcast, Sir Andrew Likierman, professor of management practice in accounting at explains the key to making the right calls https://t.co/hshYetHDLW https://t.co/Rf4km91yjo',
     'With lockdowns being relaxed in many parts of the world, what will be the long-term effects of the coronavirus pandemic on public transport? asks author and transport guru https://t.co/HppIPgmWMH',
     'Uniqloâ\x80\x99s clothes have been called basic, bland and boring. Why is it so successful? https://t.co/xkF1lCAafR From ',
     'Rather than retreat into zero-sum identity politics, Americans yearning for progress on race should look to Enlightenment liberalism https://t.co/FwdHBvQYDp',
     'On the latest episode of â\x80\x9cThe World Aheadâ\x80\x9d podcast:\n\n- The future of public transport in the wake of the pandemic\n- on how countries should prepare for future disasters\n- Could a â\x80\x9ccarbon surveillanceâ\x80\x9d system help save the planet? \n\nhttps://t.co/zKdHEYI4ng',
     'The horseshoe bats in Yunnan province which harbour close relatives of SARS-CoV-2 are found across South-East Asia https://t.co/llVeihGX93',
     'Donald Trumps decision to send federal agents into American cities, for rule-of-law implications alone, is one of his most reckless moves yet https://t.co/XAJx614RTI',
     'Pandemic-driven insomnia has led to a boom in bedtime stories...for adults. tells â\x80\x9cThe Intelligenceâ\x80\x9d she isnâ\x80\x99t too fond of the one read by John McEnroe https://t.co/mvZkGy8jIr https://t.co/7B0x9hn5OY',
     'A profound shift is taking place in economics: governments can now spend as they please. That presents opportunitiesâ\x80\x94and grave dangers. Our cover this week https://t.co/HZmdIa7KHl https://t.co/Aw7ZhNTM4m',
     'Weve updated the Big Mac index https://t.co/7WQld5ynlf',
     'In this new era of masked encounters, a new kind of social acumen is required, writes Pamela Druckerman in https://t.co/HmBQwljO3y',
     'On this weekâ\x80\x99s â\x80\x9cMoney Talksâ\x80\x9d podcast:\n\n-How can TikTok navigate tensions between China and America?\n-Covid-19 has left some households flush with cash, but theyâ\x80\x99re in no rush to spend it\n-And, the key to making better judgments in business \n\nhttps://t.co/7iK4AYsLGP',
     'Israelâ\x80\x99s prime minister has been loudly boasting of his covid-19 successes. Now, tells â\x80\x9cThe Intelligenceâ\x80\x9d, cases are multiplyingâ\x80\x94as are protests https://t.co/De0vDcg2Cj',
     'The world is full of people whose lack of judgment brought their careers or personal life crashing down https://t.co/EPFGlHL9SS',
     'Donald Trumps decision to send federal agents into American cities, for rule-of-law implications alone, is one of his most reckless moves yet https://t.co/tWmH22oH6h',
     'The Islamic world inspired Western artists for centuries https://t.co/EhD2ycBwqt From the archive',
     'Early and extreme seasonal floods have inundated China. On â\x80\x9cThe Intelligenceâ\x80\x9d says that raises questions about the countryâ\x80\x99s river-management promises https://t.co/dCSjMbm6bO https://t.co/6v0jgQbR5i',
     'Today on â\x80\x9cThe Intelligenceâ\x80\x9d: Israelâ\x80\x99s covid-19 spike brings protests, extreme flooding in China threatens millions and a boom in bedtime stories...for adults https://t.co/9q3ELNdyoN',
     'Its unlikely a replacement stimulus package will be agreed upon by the end of the month, resulting in a gap that could upend the livelihood of millions of Americans https://t.co/QbI4hZh6Nt',
     'Hong Kongs new national-security law is one of the biggest assaults on a liberal society since the second world war https://t.co/gAQY9lAVpA',
     'In need of rebalancing the polls, the president has deployed immigration law enforcers to American cities https://t.co/ba1Qodz9tY',
     'TikTok has been downloaded nearly 2bn times worldwide. Now its at risk of being banned in America. On â\x80\x9cMoney Talksâ\x80\x9d, explains what the Chinese-owned firm is doing to address concerns over privacy and security https://t.co/R1T1n4XPai',
     'The Big Mac index makes exchange-rate theory more digestible https://t.co/LTGYuk87jZ',
     'Yiyun Li: â\x80\x9cI read "War and Peace" every year. Each time I see new things" https://t.co/fkrh53dBdg From ',
     'Irelands policy of neutrality helps it avoid unpopular military entanglement. But it is in the EU where it shows true diplomatic dexterity https://t.co/eIyariYEpi',
     'If first-time online shoppers in Latin America make it a habit, MercadoLibre has plenty to gain https://t.co/Af2PnhB8Lm',
     'â\x80\x9cLoving and Letting Goâ\x80\x9d, an exhibition in Cologne, looks at how the artist drew on her own sorrow to give her work its emotive power https://t.co/R30S0oLGHC',
     'The collective dancing carried on, unexplained, for two months. Distressed and in pain, those afflicted seemed to be moving against their will https://t.co/2CDbOU7Jt5',
     'The world as we used to know it lives on in Google Maps https://t.co/doJQ2mAr4R From ',
     'Fewer shopping, dining and holiday options means many households in the rich world are saving money. On â\x80\x9cMoney Talksâ\x80\x9d, Britain economics correspondent says the easing of lockdowns wonâ\x80\x99t result in a rush to splash the cash https://t.co/cA6mkWcLz8 https://t.co/X9DC4MAywh',
     'The coronavirus pandemic has forced many people to make difficult judgments. On our latest â\x80\x9cMoney Talksâ\x80\x9d podcast, Sir Andrew Likierman, professor of management practice in accounting at explains the key to making the right calls https://t.co/lPLDTDQ4k3 https://t.co/GHgjipAwoU',
     'Sign up to The Climate Issue to receive insightful climate-change analysis for free in your inbox every fortnight  https://t.co/KgIbJtRr9U',
     'The lockdown has produced two types of workers: the slackers and the Stakhanovites. Which one are you? https://t.co/4CVD3haEZi',
     'Software programmers make between 0.5 and 50 errors in every 1,000 lines of code they write https://t.co/qF4cB6DPjz',
     'What was the point of offices anyway? https://t.co/pFv8f2KGK0 From ',
     'How has the pandemic shaped Americas presidential election so far? Sign up to Checks and Balance, our weekly newsletter on US politics, for fair-minded analysis of the campaign https://t.co/Q39nfYWMnx',
     'On this weekâ\x80\x99s â\x80\x9cMoney Talksâ\x80\x9d podcast:\n\n-How can TikTok navigate tensions between China and America?\n-Covid-19 has left some households flush with cash, but theyâ\x80\x99re in no rush to spend it\n-And, the key to making better judgments in business \n\nhttps://t.co/wkwpH2YHob',
     'With about 18m still unemployed, compared with 6m before the recession, Americaâ\x80\x99s hope for a â\x80\x9cV-shaped recoveryâ\x80\x9d seems out of reach https://t.co/o06ggI2wk7',
     'Some South-East Asian countries are likely to be an evolutionary hotspot for coronavirusesâ\x80\x94one that mirrors bat diversity https://t.co/vVifob7vd0',
     'Two years from now, will a second wave of the pandemic have wiped out air travel? https://t.co/PDvPOCKhHV',
     'The famous â\x80\x9cKeep Calm and Carry Onâ\x80\x9d slogan was never deployed in wartime: reports on the morale of civilians pointed to boredom, not panic  https://t.co/eE65vAZsxd From ',
     'The world is full of people whose lack of judgment brought their careers or personal life crashing down https://t.co/qCShDkwM2R',
     'How could carbon-removal technologies change the world order? An imagined scenario from 2050 https://t.co/0uK2N0knVi',
     'Studies suggest that under-18s are a third to a half less likely to catch covid-19. This bolsters the argument for reopening schools https://t.co/vLu0vR9Kii',
     'Casual sex is out, companionship is in https://t.co/nXg3VckzHg',
     'The end of the grand fantasy: restaurant dining may never be the same again https://t.co/9GQLRg3xVn From ',
     'What was the point of offices anyway? https://t.co/Y9bgswTva6 From ',
     'OnlyFans is leading a revolution in sex work, giving creators total control of their image and workload https://t.co/mLVOzUiVIk From ',
     'The consummate moderate now has a strong chance of becoming presidentâ\x80\x93â\x80\x93and with the most ambitious platform of any Democratic candidate in generations https://t.co/oOlymQLOkT',
     'Measuring luminescence helps to date a remarkable new discovery at Stonehenge https://t.co/YFPjnJrGzi',
     'Hong Kongs new national-security law is one of the biggest assaults on a liberal society since the second world war https://t.co/RVI4d2PWif',
     'Why is a mattress company called Casper? Where did Slack get its name? The secrets behind popular brand names https://t.co/cAyPCR6Da4 From ',
     'Uniqloâ\x80\x99s clothes have been called basic, bland and boring. Why is it so successful? https://t.co/VE5FTD9vo4 From ',
     'How has covid-19 taken advantage of distrust in America? Sign up to Checks and Balance, our weekly newsletter devoted to US politics, to learn more https://t.co/QTHihxJVmd',
     'Americaâ\x80\x99s stimulus package included expiration dates that are fast approaching. Democrats and Republicans cant agree on what another might look like https://t.co/BJVsBXm4ph',
     'The Three Gorges Dam has failed to protect China in its struggle against devastating floods, despite suggestions that it could https://t.co/C5t7nTcOej',
     'Weve updated the Big Mac index https://t.co/EMtTFcXTrU',
     'Four hundred years on, Artemisia Gentileschiâ\x80\x98s paintings resonate more than ever https://t.co/TwuG3vTUTE From ',
     'Some think the origins of the virus are not to be found in China at all, but rather just across the border in Myanmar, Laos or Vietnam https://t.co/QWYBe9OjE7',
     'Most Chileans agree that the state should act to reduce inequality. But their anger could create support for populist policies that would make the country poorer https://t.co/bQAQLzFakw',
     'Perseverance, which cost $2.4bn to build and will take another $300m to operate, will be the most sophisticated rover ever sent by America to Mars https://t.co/k8QNZfYYz7',
     'How do you get someone to order $100,000 of drinks at a nightclub? https://t.co/rsDYAYsNyQ From ',
     'Listen to â\x80\x9cEditorâ\x80\x99s Picksâ\x80\x9d to hear essential stories from The Economist, read aloud. This week: Huawei and the tech cold war, reopening classrooms in a crisis and new transistor technology https://t.co/4xRl1X2Syd',
     'Two planes from the same airline crashed in the same spot in the Alps, 16 years apart. Now the melting ice is revealing their secrets https://t.co/6eIL6UTi5W From ',
     'You may be exhausted but the covid-19 pandemic is barely getting started https://t.co/03yY1fQnhS',
     'By 2040 the worlds natural wealth is predicted to decline by a fifth  https://t.co/jfNtEfq5Ko',
     'KALâ\x80\x99s cartoon https://t.co/pQOsYNYbkL',
     'South Africaâ\x80\x99s social-security system was supposed to cushion the blow of the pandemic, but it has been woefully mismanaged https://t.co/d53l1YDbRX',
     'The brightest comet in 25 years is lighting up the northern hemisphereâ\x80\x99s night sky. On â\x80\x9cThe Intelligenceâ\x80\x9d we speak to the scientist who discovered it in March https://t.co/YFGdthYQVQ https://t.co/mBrbPmr3fg',
     'Countries that have succeeded in â\x80\x9cflattening the curveâ\x80\x9d of the virus have often enacted mandatory mask policies alongside lockdowns https://t.co/rJOUZtX2k1',
     'When it comes to beliefs about covid-19, a new survey suggests that foolhardiness comes with age, and prudence with youth https://t.co/9YYjIPGgLY',
     'One way of testing if money buys happiness is by analysing what happens when it disappears https://t.co/rAOzvBvcSY',
     'â\x80\x9cBefore we know it, we had ceded many parts of the South.â\x80\x9d Americans from north-eastern states vacationing for Memorial Day weekend could have caused coronavirus flare-ups in the South, tells on â\x80\x9cThe Economist Asksâ\x80\x9d https://t.co/snbVQUjyIu',
     'â\x80\x9cI was dancing and hating my lifeâ\x80\x9d. Welcome to the parties of the super-rich https://t.co/BSzcSAkevO From ',
     'Many brands donâ\x80\x99t have a history. So now theyâ\x80\x99re inventing them https://t.co/X5QzNYeDbD From ',
     'South Africas social-security system was supposed to help 15m unemployed people. By early June just 600,000 had been paid https://t.co/XdoH918RzZ',
     'British authorities donâ\x80\x99t have hard proof of Russian meddling in institutions and electionsâ\x80\x94because, tells â\x80\x9cThe Intelligenceâ\x80\x9d, they havenâ\x80\x99t looked https://t.co/dW3KCF9ygr',
     'The insurance industry is scrambling to find ways to be helpful ahead of the next shock https://t.co/riPZBO5f73',
     'You absorb fewer calories eating toast that has been left to go cold, or leftover spaghetti, than if they were freshly made https://t.co/YXBLvvTmcb From the archive archive',
     'Today on â\x80\x9cThe Intelligenceâ\x80\x9d: Britainâ\x80\x99s troubling report into Russian meddling, tackling Myanmarâ\x80\x99s meth-addiction mess and a bright prospect for comet-spotters https://t.co/TacNfMHWXB',
     'Should Germany choose to ban Huawei in the autumn, a corner will have been turned. America will have used its might to humble one of Chinas national champions https://t.co/KQ0ySqu4eY',
     'Shanghaiâ\x80\x99s STAR market, one year old today, has done exceptionally well https://t.co/zVbSB8RoMX',
     'Stay up to speed with our latest climate coverage by signing up for The Climate Issue newsletter, which brings the best of our journalism to you every fortnight https://t.co/eJBe2jHqKG',
     'On â\x80\x9cThe World Aheadâ\x80\x9d podcast, and imagine the rise of a â\x80\x9ccarbon surveillanceâ\x80\x9d system that tracks everyoneâ\x80\x99s emissions https://t.co/o7xcGFgYGq',
     'â\x80\x9cI think it is really important to get these schools reopened, not as a father and a grandfather of 11, but I think the public health...is not served by having schools closed.â\x80\x9d Robert Redfield is the latest guest on â\x80\x9cThe Economist Asksâ\x80\x9d https://t.co/PwD2v8sbmx',
     'Listen to â\x80\x9cEditorâ\x80\x99s Picksâ\x80\x9d to hear essential stories from The Economist, read aloud. This week: Huawei and the tech cold war, reopening classrooms in a crisis and new transistor technology https://t.co/ywND6RkyeN',
     'â\x80\x9cAmerica looks a lot worse...relative to Europe.â\x80\x9d On â\x80\x9cChecks and Balanceâ\x80\x9d examines the extraordinary surge of covid-19 cases in the United States https://t.co/C6IYmD069J https://t.co/EB1sA7DTFz',
     'Struggling to stay up to speed with Americas presidential race? Sign up to Checks and Balanceâ\x80\x94our weekly newsletter on US politicsâ\x80\x94for rigorous, fair-minded coverage of the campaign https://t.co/CSxQSjobHS',
     'For decades, Myanmar supplied heroin to the world. Now, tells â\x80\x9cThe Intelligenceâ\x80\x9d, itâ\x80\x99s a methamphetamine powerhouse with its own addiction issues https://t.co/mZDgGxSxWT https://t.co/jrMkkL4GE5',
     'What effect has covid-19 had on climate change? Sign up for The Climate Issue newsletter to receive the best of our analysis on environmental matters every fortnight https://t.co/7aLdLGFaX1',
     'On â\x80\x9cBabbageâ\x80\x9d:\n\n-Professor Sarah Gilbert of on the timeline for an effective covid-19 vaccine\n-Navigating the sky with diamonds\n-And why sewage is helping census-takers \n\nhttps://t.co/2kWb3tkPAn',
     'Sign up to The Climate Issue to receive insightful climate-change analysis for free in your inbox every fortnight  https://t.co/6kLtlr3Wi9',
     'â\x80\x9cBefore we know it, we had ceded many parts of the South.â\x80\x9d Americans from north-eastern states vacationing for Memorial Day weekend could have caused coronavirus flare-ups in the South, tells on â\x80\x9cThe Economist Asksâ\x80\x9d https://t.co/mAUzUVXudx',
     'Can we escape from information overload? Ask the man who spent a month living in in the dark https://t.co/p1pwaIutQC From ',
     'â\x80\x9cThe government has been spraying large amounts of cash at the effort.â\x80\x9d tells that an American vaccine is possible in time for election day, on â\x80\x9cChecks and Balanceâ\x80\x9d https://t.co/KKLqgVEojY',
     'Which states will Donald Trump win come November? Which ones are likely to turn blue? Our new statistical model is free to read https://t.co/OieX2fAQZM',
     'â\x80\x9cI would assume that were going to have somewhere hopefully between one and three vaccines approved or human use prior to January.â\x80\x9d Robert Redfield is the latest guest on â\x80\x9cThe Economist Asksâ\x80\x9d podcast https://t.co/MECHITXdpJ https://t.co/RiSXmJ4mS8',
     'Twitter has announced a ban on over 7,000 QAnon accounts. What happens when you lose a loved one to the QAnon conspiracy? Last month found out https://t.co/WdTPf9Rvy0',
     'British authorities donâ\x80\x99t have hard proof of Russian meddling in institutions and electionsâ\x80\x94because, tells â\x80\x9cThe Intelligenceâ\x80\x9d, they havenâ\x80\x99t looked https://t.co/9OqfHnTL6C',
     'The agreement that the 27 countries will borrow and spend vast sums collectively comes as good news for advocates of a stronger EU https://t.co/bU5Bqcysvm',
     'The worlds economies have been moving away from oil for some time. Covid-19 has simply accelerated the shift https://t.co/6KmSrgw8CO',
     'Fewer shopping, dining and holiday options means many households in the rich world are saving money. On â\x80\x9cMoney Talksâ\x80\x9d, Britain economics correspondent says the easing of lockdowns wonâ\x80\x99t result in a rush to splash the cash https://t.co/3uB0EzGAEz https://t.co/ZhvNmjtBGo',
     'Subscribe to The Economist for 12 weeks access with our introductory offer and enjoy a fresh perspective on the issues shaping our world https://t.co/fuR9dP315L https://t.co/TcpIZbNIll',
     'On this weekâ\x80\x99s â\x80\x9cMoney Talksâ\x80\x9d podcast:\n\n-How can TikTok navigate tensions between China and America?\n-Covid-19 has left some households flush with cash, but theyâ\x80\x99re in no rush to spend it\n-And, the key to making better judgments in business \n\nhttps://t.co/1oP2DU0BLX',
     'With lockdowns being relaxed in many parts of the world, what will be the long-term effects of the coronavirus pandemic on public transport? asks author and transport guru https://t.co/vNHGnbg4wy',
     'The United Arab Emirates has joined the club of countries that have launched probes towards extraterrestrial bodies https://t.co/PnTNNDw7hu',
     'The brightest comet in 25 years is lighting up the northern hemisphereâ\x80\x99s night sky. On â\x80\x9cThe Intelligenceâ\x80\x9d we speak to the scientist who discovered it in March https://t.co/od9AoveEOI https://t.co/hrtnpnvzM1',
     'Today on â\x80\x9cThe Intelligenceâ\x80\x9d: Britainâ\x80\x99s troubling report into Russian meddling, tackling Myanmarâ\x80\x99s meth-addiction mess and a bright prospect for comet-spotters https://t.co/yfyxkTzrgQ',
     'Chris Corbin and Jeremy King know why people go to restaurants. Hint: itâ\x80\x99s not about the food https://t.co/qnKREQPSxf From ',
     'If Arab rulers want their citizens to start earning a living, they will need to start earning the consent of the ruled https://t.co/xpCjskUvrF',
     'Listen to â\x80\x9cEditorâ\x80\x99s Picksâ\x80\x9d to hear essential stories from The Economist, read aloud. This week: Huawei and the tech cold war, reopening classrooms in a crisis and new transistor technology https://t.co/0N3x7VFQ6Q',
     'For decades, Myanmar supplied heroin to the world. Now, tells â\x80\x9cThe Intelligenceâ\x80\x9d, itâ\x80\x99s a methamphetamine powerhouse with its own addiction issues https://t.co/PNwSttXFLk https://t.co/Gk6GOTB2LD',
     'When clear rules are ignored by officers, their wilful failure can be the basis for their criminal liability, argues Franklin Zimring https://t.co/WvCcl28zjo',
     'When will it be safe to reopen schools in America? Robert Redfield says â\x80\x9cIts not opening schools versus public health. Its public health versus public health,â\x80\x9d on the latest episode of â\x80\x9cThe Economist Asksâ\x80\x9d https://t.co/HxDvLN2UrV',
     'The coronavirus pandemic has forced many people to make difficult judgments. On our latest â\x80\x9cMoney Talksâ\x80\x9d podcast, Sir Andrew Likierman, professor of management practice in accounting at explains the key to making the right calls https://t.co/U6hDdV5ZY1 https://t.co/Cx3yUwwqFa',
     'Lee Child has no regrets about retiring. He doesnâ\x80\x99t even like Jack Reacher that much https://t.co/AnMv03azhC From ',
     'â\x80\x9cIf everybody...would just put on a face covering...the outbreak in America would be over.â\x80\x9d says the science is now clear on our latest â\x80\x9cChecks and Balanceâ\x80\x9d podcast https://t.co/7jYrW7rJSt',
     'What are the biggest challenges facing Donald Trump and Joe Biden ahead of the election in November? Subscribe to Checks and Balance, our weekly newsletter on American politics, for rigorous analysis of the campaign https://t.co/23e19Zcm6T',
     'On the latest episode of â\x80\x9cThe World Aheadâ\x80\x9d:\n\n- The future of public transport in the wake of the pandemic\n- on how countries should prepare for future disasters\n- Could a â\x80\x9ccarbon surveillanceâ\x80\x9d system help save the planet? \n\nhttps://t.co/M7f2Yhz8NO',
     'The brightest comet in 25 years is lighting up the northern hemisphereâ\x80\x99s night sky. On â\x80\x9cThe Intelligenceâ\x80\x9d we speak to the scientist who discovered it in March https://t.co/k6ZbJC5qYY https://t.co/sxYZMPKNma',
     'â\x80\x9cWho is this person? Sheâ\x80\x99s not the same mother I used to have, and probably never will be again.â\x80\x9d From https://t.co/JmVByyiIiD',
     'Tommaso Buscetta probably had a bigger impact than any supergrass in history https://t.co/MiA5eXWdP7',
     'Listen to â\x80\x9cEditorâ\x80\x99s Picksâ\x80\x9d to hear essential stories from The Economist, read aloud. This week: Huawei and the tech cold war, reopening classrooms in a crisis and new transistor technology https://t.co/TlCZ79X2BJ',
     'The deal falls some way short of the â\x80\x9cHamiltonian momentâ\x80\x9d some had hoped for it https://t.co/yJR10zcvCh',
     'British authorities donâ\x80\x99t have hard proof of Russian meddling in institutions and electionsâ\x80\x94because, tells â\x80\x9cThe Intelligenceâ\x80\x9d, they havenâ\x80\x99t looked https://t.co/kCyOsqdIEw',
     'Twitter is cracking down on content and accounts related to QAnon. How did the conspiracy theory gain so much momentum in the first place? Last month found out https://t.co/PbvRUSfIXR',
     'The programme is equivalent to 4.7% of the EUâ\x80\x99s GDP https://t.co/CD8jDxF0yC',
     'For decades, Myanmar supplied heroin to the world. Now, tells â\x80\x9cThe Intelligenceâ\x80\x9d, itâ\x80\x99s a methamphetamine powerhouse with its own addiction issues https://t.co/2k6jI8ZPj5 https://t.co/6E58XT2LQ3',
     'Today on â\x80\x9cThe Intelligenceâ\x80\x9d: Britainâ\x80\x99s troubling report into Russian meddling, tackling Myanmarâ\x80\x99s meth-addiction mess and a bright prospect for comet-spotters https://t.co/aI5eg8XR7u',
     'We are finally living out the "Keep Calm and Carry On" fantasy. It isnt all its cracked up to be https://t.co/p2jIYYo2Hs From ',
     'TikTok has been downloaded nearly 2bn times worldwide. Now its at risk of being banned in America. On â\x80\x9cMoney Talksâ\x80\x9d, explains what the Chinese-owned firm is doing to address concerns over privacy and security https://t.co/XygbPF0Mvi',
     'â\x80\x9cWhere a lot of these hotspots started...the southern individuals who had been spared from this outbreak really werent embracing the social distancing strategies.â\x80\x9d on coronavirus flare-ups in Americaâ\x80\x99s southern states, on â\x80\x9cThe Economist Asksâ\x80\x9d https://t.co/ANIEvbLwQk https://t.co/rmCCKMilzX',
     'Sign up to The Climate Issue newsletter to receive the best of our climate-change analysis, delivered to you every fortnight https://t.co/cacqqZwS0O',
     'South Africas alcohol ban has shot pineapple prices through the roofâ\x80\x94why? https://t.co/YPxwMMFK9H',
     'On the latest episode of â\x80\x9cThe World Aheadâ\x80\x9d:\n\n- The future of public transport in the wake of the pandemic\n- on how countries should prepare for future disasters\n- Could a â\x80\x9ccarbon surveillanceâ\x80\x9d system help save the planet? \n\nhttps://t.co/018QUZxopW',
     'â\x80\x9cIt has so quickly become another battle in the culture wars.â\x80\x9d tells our â\x80\x9cChecks and Balanceâ\x80\x9d podcast why America is doing so badly compared to Europe in restraining covid-19 https://t.co/vSDygG0Zmu https://t.co/DqWYk7HalD',
     'Around 3.5bn years ago conditions on Earth and Mars were similar. Might small traces of life have survived on the freezing desert of Mars? https://t.co/lIiaHonBEu',
     'The battle against coronavirus resembles not the front line but the Home Front https://t.co/dq73euPYjt From ',
     'Younger Americans surveyed consider themselves more likely to catch covid-19, and to die from it, than older respondents do https://t.co/QkK4w6Jxli',
     'What effect has covid-19 had on climate change? Sign up for The Climate Issue newsletter to receive the best of our analysis on environmental matters every fortnight https://t.co/osRiE2XHZA',
     'The simple life doesnâ\x80\x99t mean fun. It means an awful lot of washing up https://t.co/OfD1IJUaW9 From ',
     'The result fell short of what some had hoped for. But it was greeted as a triumph in Paris and Berlin, and by believers in a more powerful and more federal EU https://t.co/1HQcRKTCaF',
     'The EU has now marshalled a fiscal response to the covid crisis equal to or better than Americaâ\x80\x99s https://t.co/gJnPOdXMUw',
     'Companies spend $10,000 on office space per employee every year, on average. Has the pandemic turned the office into an expensive artefact? https://t.co/ADMB71ySe1 https://t.co/CmNFfgjXvK',
     'On the latest episode of â\x80\x9cThe World Aheadâ\x80\x9d:\n\n- The future of public transport in the wake of the pandemic\n- on how countries should prepare for future disasters\n- Could a â\x80\x9ccarbon surveillanceâ\x80\x9d system help save the planet? \n\nhttps://t.co/75h1dGy4un',
     'In the end, these seemingly principled disputes were resolved through old-fashioned horse-trading https://t.co/JKVwbs4Xrq',
     'A recent survey suggests Americans aged 18-34 consider themselves to be nearly three times more likely to contract covid-19 than respondents over 70 do https://t.co/CjNVMhQ9Rr',
     'Exchange rates move rapidly, but prices are sticky. In the Big Mac index, both matter https://t.co/4lVYjRFNso',
     'On average, countries with more natural capital also tend to have a higher GDP per person https://t.co/RbNsoB7cmE',
     'â\x80\x9cIt has so quickly become another battle in the culture wars.â\x80\x9d tells our â\x80\x9cChecks and Balanceâ\x80\x9d podcast why America is doing so badly compared to Europe in restraining covid-19 https://t.co/szgpSE2OjK https://t.co/3lXpSvLwms',
     'No Arab oil producer, except Qatar, will be able to balance its budget at the current oil price of roughly $40 a barrel https://t.co/CX7v1CUYbf',
     'Vietnam is a go-to place for production that has become too costly in China. But there is a catch https://t.co/AtsFckwFPW',
     'Writing in The Economist, Franklin Zimring explains how practical policies could substantially reduce civilian deaths at the hands of the police https://t.co/QbEdNvFOcy',
     'The final compromise was not too distant from France and Germanyâ\x80\x99s initial proposal, which laid the groundwork for the deal https://t.co/S7JRrcyZcV',
     'Sign up to The Climate Issue newsletter to receive the best of our climate-change analysis, delivered to you every fortnight https://t.co/1GGb5n4Owt',
     'At the March on Washington in 1963, John Lewis vowed to â\x80\x9csplinter the segregated South into a thousand pieces and put them back together in the image of God and democracyâ\x80\x9d https://t.co/Zakp6sVW7y',
     'Young people are no more cavalier about the coronavirus than their older counterparts https://t.co/e1K1BhjQ4K',
     'Most historians believe that the â\x80\x9cdancing plagueâ\x80\x9d was in fact a mass psychogenic illness, often referred to as â\x80\x9cmass hysteriaâ\x80\x9d https://t.co/CjSp4XypgK',
     'On â\x80\x9cBabbageâ\x80\x9d:\n-How is covid-19 transmitted in the air, and should ventilation be mandated in public places?\nof on how she would fix Americas healthcare system\n-And the illuminating technology revealing archaeological secrets \n\nhttps://t.co/YdWOeCHoH3',
     'Like his hero Jack Reacher, Lee Child is a lone ranger https://t.co/MhsfKd9Edp From ',
     'On â\x80\x9cThe Intelligenceâ\x80\x9d says that a charity in Burkina Faso is reducing the stigma for prison inmatesâ\x80\x94by turning them into pop stars https://t.co/a06pxFFEEJ https://t.co/3XT8WVSpZE',
     'The package proposed by Angela Merkel and Emmanuel Macron faced opposition from wealthy northern countries and the populist leaders of Hungary and Poland https://t.co/oQWX5D3Yb8',
     'â\x80\x9cWhere a lot of these hotspots started...the southern individuals who had been spared from this outbreak really werent embracing the social distancing strategies.â\x80\x9d on coronavirus flare-ups in Americaâ\x80\x99s southern states, on â\x80\x9cThe Economist Asksâ\x80\x9d https://t.co/2JaZU06SpQ https://t.co/gNkefsyxKC',
     'â\x80\x9cIt has so quickly become another battle in the culture wars.â\x80\x9d tells our â\x80\x9cChecks and Balanceâ\x80\x9d podcast why America is doing so badly compared to Europe in restraining covid-19 https://t.co/nTHxtnwKvu https://t.co/99cFrLx7GY',
     'Listen to â\x80\x9cEditorâ\x80\x99s Picksâ\x80\x9d to hear essential stories from The Economist, read aloud. This week: Huawei and the tech cold war, reopening classrooms in a crisis and new transistor technology https://t.co/xr310Zbo4O',
     'How far and how long does covid-19 linger in the air? Professor Lidia Morawska says ventilation should be mandated in public places. Hear more on â\x80\x9cBabbageâ\x80\x9d https://t.co/c7Z3OKh9nq https://t.co/qz8wvGWWt7',
     'The deal struck by the EUâ\x80\x99s 27 national leaders has two elements: the regular EU budget and a â\x82¬750bn covid-19 recovery fund  https://t.co/N3Bpe6iXzf',
     'Europeâ\x80\x99s leaders have at last agreed to a recovery dealâ\x80\x94and tells â\x80\x9cThe Intelligenceâ\x80\x9d it includes once-unthinkable provisions for collective debt https://t.co/zB5LpQ7WP5',
     'What was the point of offices anyway? https://t.co/yCq83z6CUM From ',
     'What would Donald Trump do with another term in office? For rigorous, fair-minded coverage of the election, sign up for Checks and Balance, our weekly newsletter devoted to American politics https://t.co/16V5PK1KWh',
     'From 1990 to 2014, the stock of natural capital per person fell in 128 out of 140 countries https://t.co/Zc9u9Cr2rW',
     'South Africa is trying to balance surging covid-19 cases with rekindling economic activity https://t.co/0DoK4AICuj',
     'The famous â\x80\x9cKeep Calm and Carry Onâ\x80\x9d slogan was never deployed in wartime: reports on the morale of civilians pointed to boredom, not panic  https://t.co/9jdFtX5ypd From ',
     'Today on â\x80\x9cThe Intelligenceâ\x80\x9d: Europeâ\x80\x99s landmark collective-debt recovery deal, jihadism spills out of the Sahel and how Burkina Fasoâ\x80\x99s inmates turn into pop stars https://t.co/IIBnbUxPDy',
     'Writing in The Economist, Franklin Zimring explains how "dont shoot" rules can remove the supposed justification for lethal force in whole categories of encounters between citizens and the police https://t.co/p5iVE1DDX5',
     'Jihadism has long been growing in the Sahel region. Now, tells â\x80\x9cThe Intelligenceâ\x80\x9d insurgents are spilling out toward Africaâ\x80\x99s most populous states https://t.co/DRAGFzr0mO https://t.co/JY2oRUxvOI',
     'The hard-fought deal shows that the blocâ\x80\x99s members have the sense of solidarity needed to respond collectively to disasters, despite internal political splits https://t.co/sw83dtXaNO',
     'John Lewis never stopped believing in Martin Luther Kingâ\x80\x99s Beloved Community, centred on radical love and justice. He never stopped making good trouble https://t.co/3bn4NRyeho',
     'On this weekâ\x80\x99s â\x80\x9cMoney Talksâ\x80\x9d podcast:\n\n-How can TikTok navigate tensions between China and America?\n-Covid-19 has left some households flush with cash, but theyâ\x80\x99re in no rush to spend it\n-And, the key to making better judgments in business \n\nhttps://t.co/gqhiViqBJI',
     'When it comes to style, a streak of scarlet will never go out of fashion https://t.co/PZ762Kp1gf From ',
     'On â\x80\x9cThe Intelligenceâ\x80\x9d says that a charity in Burkina Faso is reducing the stigma for prison inmatesâ\x80\x94by turning them into pop stars https://t.co/2X9J5Mnj8M https://t.co/k6sZItdfDH',
     'Over two-thirds of Africans say democracy is the best form of government. But they are frustrated with it  https://t.co/OJbmOwfbYi',
     'South Africas alcohol ban has shot pineapple prices through the roofâ\x80\x94why? https://t.co/hJBdvbj3UY',
     'Europeâ\x80\x99s leaders have at last agreed to a recovery dealâ\x80\x94and tells â\x80\x9cThe Intelligenceâ\x80\x9d it includes once-unthinkable provisions for collective debt https://t.co/r3OadzmpC1',
     'Four of Americas largest banks are sitting on $50bn of bad-debt reserves, exceeding the amount set aside at the height of the global financial crisis https://t.co/QJtbjBEf2t',
     'How far and how long does covid-19 linger in the air? Professor Lidia Morawska says ventilation should be mandated in public places. Hear more on â\x80\x9cBabbageâ\x80\x9d https://t.co/hVS4qb8NFL https://t.co/jEte38kyuY',
     'Many voters view Singapores ruling party as arrogant and elitist. Promises to spruce up public housing fell flat too https://t.co/wGJtMX7XKz',
     'The end of the grand fantasy: restaurant dining may never be the same again https://t.co/slnGZ8etEM From ',
     'Today on â\x80\x9cThe Intelligenceâ\x80\x9d: Europeâ\x80\x99s landmark collective-debt recovery deal, jihadism spills out of the Sahel and how Burkina Fasoâ\x80\x99s inmates turn into pop stars https://t.co/G8zbOS0c2d',
     'The EU is known for its long, gruelling summits. But its most recent was one for the history books https://t.co/DRiH8bXnTg',
     'Investment banking has had an exceptionally good year. But provisions for expected losses have been exceptionally costly https://t.co/0bqHx3Jvqb',
     'The risks of keeping schools closed far outweigh the benefits. Listen to â\x80\x9cEditorâ\x80\x99s Picksâ\x80\x9d to hear essential stories from the latest issue of The Economist, read aloud https://t.co/jjtgu4QynK',
     'The number of women killed in Mexico in the first six months of this year has sharply increased, data shows. In March, we wrote about how Latin America tackles femicide https://t.co/2ruC44Wtm1',
     'We are finally living out the "Keep Calm and Carry On" fantasy. It isnt all its cracked up to be https://t.co/I9cJ2g1lzA From ',
     'Jihadism has long been growing in the Sahel region. Now, tells â\x80\x9cThe Intelligenceâ\x80\x9d insurgents are spilling out toward Africaâ\x80\x99s most populous states https://t.co/RhYUalVFub https://t.co/oOg7aahMZq',
     'Western governments have rejected Zimbabwes pleas for help from the IMF and World Bank, largely on grounds of human-rights abuses https://t.co/a0XvybbBl1',
     'On â\x80\x9cBabbageâ\x80\x9d:\n-How is covid-19 transmitted in the air, and should ventilation be mandated in public places?\nof on how she would fix Americas healthcare system\n-And the illuminating technology revealing archaeological secrets \n\nhttps://t.co/874GYdrVdz',
     'Lee Child has no regrets about retiring. He doesnâ\x80\x99t even like Jack Reacher that much https://t.co/4I9Jsf0P3Z From ',
     'Now that we are thrown back on our own resources, the â\x80\x9cmake do and mendâ\x80\x9d spirit of wartime seems less appealing https://t.co/TeGWf5ca63 From ',
     'For more than 400 years, Stonehenge has stood on Salisbury Plain in Britain. Now, a new tool is revealing hidden secrets and rewriting the archaeology of its landscape. On â\x80\x9cBabbageâ\x80\x9d Dr Tim Kinnaird explains how https://t.co/vVRUSvPZvA https://t.co/c45aeiVcYx',
     'Arab leaders are asking their people to sacrificeâ\x80\x94and giving them little say in the matter. That is a recipe for unrest https://t.co/7w6m9ETntL',
     'Young Africans want more democracy https://t.co/qi1RIBaDuH',
     'On â\x80\x9cThe Intelligenceâ\x80\x9d says that a charity in Burkina Faso is reducing the stigma for prison inmatesâ\x80\x94by turning them into pop stars https://t.co/zW05CkDuRO https://t.co/yi34Z8gxI5',
     'â\x80\x9cSomething happened between June 12 and June 16. All of these hotspots became activated.â\x80\x9d explains the surge in covid-19 cases across the South on our â\x80\x9cChecks and Balanceâ\x80\x9d podcast https://t.co/CEHLYxH3br',
     'At his last public appearance, John Lewis stood on the newly painted Black Lives Matter plazaâ\x80\x94a reminder of how far America had come, and how far it still has to go https://t.co/NQJsHrAYDy',
     'The era of easy oil money in the Middle East is ending. The result will be painful https://t.co/HFKDQPfqvQ',
     'There has been a record high number of murders in Mexico during the first six months of this year, including a sharp increase in the murders of women. In March, we explained why Latin America treats femicides differently from other murders https://t.co/N3UiQOfHqi"',
     'Europeâ\x80\x99s leaders have at last agreed to a recovery dealâ\x80\x94and tells â\x80\x9cThe Intelligenceâ\x80\x9d it includes once-unthinkable provisions for collective debt https://t.co/kVO6q2ySAQ',
     'Both antibodies and T-cells are generally necessary to provide immunity from covid-19. Oxford Universityâ\x80\x99s trial results suggest its vaccine generates both https://t.co/K3tkgxxY7D',
     'Thereâ\x80\x99s a hidden economy behind every $100,000 bar tab. Its currency is women https://t.co/UChtyHGt32 From ',
     'Jihadism has long been growing in the Sahel region. Now, tells â\x80\x9cThe Intelligenceâ\x80\x9d insurgents are spilling out toward Africaâ\x80\x99s most populous states https://t.co/dRIfzayjBS https://t.co/pgjN4G8pIG',
     'Today on â\x80\x9cThe Intelligenceâ\x80\x9d: Europeâ\x80\x99s landmark collective-debt recovery deal, jihadism spills out of the Sahel and how Burkina Fasoâ\x80\x99s inmates turn into pop stars https://t.co/CCrOLStmGO',
     'Why business in Hong Kong should be worried https://t.co/rprnDMRXoT',
     'â\x80\x9cI think it is really important to get these schools reopened, not as a father and a grandfather of 11, but I think the public health...is not served by having schools closed.â\x80\x9d Robert Redfield is the latest guest on â\x80\x9cThe Economist Asksâ\x80\x9d https://t.co/TR1AjGzDsB',
     'In China a crackdown on corruption has made state-owned oil companies less acquisitive amid closer scrutiny of foreign deals https://t.co/T7ctPf1S1i',
     'A new material helps transistors become vanishingly small https://t.co/sadlqURq5l',
     'How far and how long does covid-19 linger in the air? Professor Lidia Morawska says ventilation should be mandated in public places. Hear more on â\x80\x9cBabbageâ\x80\x9d https://t.co/RCiFeWVipp https://t.co/raIQrvo4W8',
     'Uniqloâ\x80\x99s clothes have been called basic, bland and boring. Why is it so successful? https://t.co/1OrZmM5yoo From ',
     'Without an agreement in place, a proliferation of digital-services taxes seem likely https://t.co/GZoBnNLM5a',
     'â\x80\x9cWhere a lot of these hotspots started...the southern individuals who had been spared from this outbreak really werent embracing the social distancing strategies.â\x80\x9d on coronavirus flare-ups in Americaâ\x80\x99s southern states, on â\x80\x9cThe Economist Asksâ\x80\x9d https://t.co/sCtXAUapH9',
     'How far and how long does covid-19 linger in the air? Professor Lidia Morawska says ventilation should be mandated in public places. Hear more on â\x80\x9cBabbageâ\x80\x9d https://t.co/AmSMmDAYnV https://t.co/1dvkcEVcUk',
     'Singapores ruling partys share of the vote sank from almost 70% at the previous election, in 2015, to 61% https://t.co/K1XzmEkp7a',
     'The world as we used to know it lives on in Google Maps https://t.co/xPzHyBuLQZ From ',
     'On â\x80\x9cBabbageâ\x80\x9d:\n-How is covid-19 transmitted in the air, and should ventilation be mandated in public places?\nof on how she would fix Americas healthcare system\n-And the illuminating technology revealing archaeological secrets \n\nhttps://t.co/ryKXCCf2GB',
     'The famous â\x80\x9cKeep Calm and Carry Onâ\x80\x9d slogan was never deployed in wartime: reports on the morale of civilians pointed to boredom, not panic  https://t.co/D8t9WTUI9j From ',
     'John Lewiss nonviolence was not soft or conciliatory; it was adamantine and confrontational, even in the face of death https://t.co/3x1EtI1B6H',
     'The announcement has added to hopes that science will provide an exit strategy for covid-19 some time this year https://t.co/WvmI6kEFnp',
     'How far and how long does covid-19 linger in the air? Professor Lidia Morawska says ventilation should be mandated in public places. Hear more on â\x80\x9cBabbageâ\x80\x9d https://t.co/BQhcdYQjdC https://t.co/J6MwXVRgf1',
     'Two planes from the same airline crashed in the same spot in the Alps, 16 years apart. Now the melting ice is revealing their secrets https://t.co/kBkVJu8YqC From ',
     'Why so many Singaporeans voted for the opposition https://t.co/4EbNYoWZS1',
     'South Africa bans alcohol sales https://t.co/OTHREW1oxm',
     'Data show the future trajectory of natural capital under a variety of scenarios https://t.co/7Ha4e93fOk',
     'Barely half of the worlds original mangrove forests remain. This is bad for the ocean and the planet.\nSupported by https://t.co/gDKIvr1kz9 https://t.co/9ZpH4HRJ7S',
     'Missing your weekly trip to IKEA? Hereâ\x80\x99s how to make Swedish meatballs at home https://t.co/UJnNfEfAg8 From ',
     'Did Queen Elizabeth approve the toppling of Australiaâ\x80\x99s government? https://t.co/KCTvLO9M4P',
     'Japan Incâ\x80\x99s IT needs a security patch https://t.co/vlfrRDlE7x',
     'Name a video game that would make an unbelievable animated movie. #WritingCommunity https://t.co/SJnfxSrHNF',
     'A Texas family remade Pee Wees Big Adventure. In my family, that makes them heroes. https://t.co/Mng1d93nBM',
     'Add a caption. Any caption. #CaptionThis #WritingCommunity itingCommunity https://t.co/a1aqqdAZDw',
     'The Trump White House doesnt give a fuck about your family. Just look at how they threat their own. https://t.co/J0xwU4Mdih',
     'This is just disgusting. https://t.co/FjrWRsGwjg',
     'Name this game. Wrong answers only. https://t.co/BLG7EsX0sB',
     'Progress. https://t.co/NXVbW7A3RC',
     'This is America. https://t.co/gxVZKK69It',
     'Add a caption. Any caption. #CaptionThis #WritingCommunity https://t.co/5ZNXe9b4B9',
     'Narcissists, psychopaths, and President Trump. https://t.co/eTXFEGfl1c',
     'Add a caption. Any caption. #CaptionThis #WritingCommunity https://t.co/z6TzNSF3uw',
     'Whats in a name?  #OperationLegend https://t.co/k75nrlQNB9',
     'Ultimately, I think critics should be more cognizant to conventions in publishing beyond an individual writers powers. Many hands and minds go into a book, and lots of work behind the scenes goes overlooked and unappreciated when it should be. #WritingCommunity',
     'That said, I know a professor might have to write a book in English and American spelling, depending on which imprint of Oxford theyre submitting to. A friend of mine did just that, so I empathize. It is A LOT of extra work for little return. #WritingCommunity',
     'Ive written books set in the US and UK. Should I have written the parts in England using English spelling. Im sure I could have, but it would have irritated my publishers, confused readers, and created more work that my editors could do without. No thank you. #WritingCommunity https://t.co/im6kDpySaY',
     'Add a caption. Any caption. #CaptionThis #WritingCommunity https://t.co/4xoFoh3dpO',
     'Lincoln approved that March, which I always felt he doesnt get enough credit for. His cabinet thought it mad, but Lincoln overruled them. He even kept a clipping on Shermans March in his walletâ\x80\x94I like to think as a reminder to himself.',
     'Weve come a long way since Forrest Gump. https://t.co/rMcBouPVzR',
     'Sherman was an unabashed white supremacist. I abhor his politics, but I sure am glad he humiliated the Confederacy.\n\nIf Lincoln could work with Sherman for five years, I can tolerate a Democratic Party that works with Kaisick for five minu',
     'One of my followers convinced me Thomas Brodie-Sangster would make a horrifyingly faithful Alex. https://t.co/QfrERedvdj',
     'RedLetterMedia made a compelling argument that Grandpa Joe was the real villain!',
     'My god... that beard!',
     'For clarity. (Thats Tywin Lannister on the left.) https://t.co/IB0uWUagBe',
     'Young Charles Dance looks like Willy Wonkas evil brother. https://t.co/2WX6WrwOJi',
     'Name this film. Wrong answers only. https://t.co/v25XcImfD0',
     'I legit think McConnell was flourishing how Republicans never objected to Obamas tan suit; just his race, party, popularity, repeated successes, and superior sex-appeal. https://t.co/hOhITujbU0',
     'Jimmy Carters signature looks like the logo for a damn successful 50s-theme restaurant. https://t.co/q0g01anyhW',
     'This is what voter disenfranchisement looks like. The only difference is, today, they skip the tests. https://t.co/8sraGGlrRv',
     'This is amazing. https://t.co/YPNn0qnMUJ',
     'This article demands a better headline. Any suggestions? https://t.co/P2w4KU1Pga #WritingCommunity',
     'This interests me more than the Legend of Korra.',
     'This could work.',
     'I like this.',
     'Pitch an Avatar spin-off in one sentence. Example: Avatar: The White Lotus, a prequel series that opens on "Book One: The Dragon of the West"  #WritingCommunity https://t.co/nXZ2RLSlOR',
     'Theres nothing like a hardcover as informative as it is handsome. A solid read. 4.5 stars. Buy a copy from your local bookstore. Full review on https://t.co/Fzq71wUE7e https://t.co/aZBmym89it',
     'Add a caption. Any caption. #RIPRobinWilliams #WritingCommunity https://t.co/F6LriYdtvQ',
     'I miss the Mediterranean. ð\x9f\x98¢ My wife and I were supposed to spend our honeymoon there. Maybe next year.',
     'So many congrats! I cant wait to read this. Im very happy for you. #WritingCommunity',
     'Name this man. Wrong answers only. https://t.co/EeVDBeGlI0',
     'Papa Johns topped with dog shit. (I.e., a Papa Johns sausage pizza.) https://t.co/TY1lXmeE8t',
     'Add a caption. Any caption. #CaptionThis #WritingCommunity https://t.co/iREPi5XM00',
     'This is big. https://t.co/XYxjNYST3R https://t.co/XYxjNYST3R',
     'Sadly, yes.',
     'This is America, and it is heartbreaking. ð\x9f\x98¢ https://t.co/JFnux1ToTS',
     'The sad thing is 70% of Trump voters probably agree with this. https://t.co/Livv45JMQa',
     'Add a caption. Any caption. #CaptionThis #WritingCommunity https://t.co/32zlkSYaqY',
     'Im shocked. Shocked. https://t.co/oKGus1GhAk',
     'Im guessing #BenGarrisons favorite Simpsons moment is a tie between George Bush spanking Bart and Ned Flanders wearing "nothing at all!" https://t.co/BTKGUrBX21',
     'Ah yes...',
     'To anyone who thinks Al Bundy is a Trump voter, heres him celebrating Obamas tax plan. https://t.co/PsbRP6Kdwj',
     'But of course they are. https://t.co/eNQfUEzBwQ',
     'Damn straight. https://t.co/pq3uBz1B7y',
     'Stop this nonsense. Hank Hill would never, ever vote for Trump. https://t.co/t2SAMXWQ6A',
     'Add a caption. Any caption. #CaptionThis #WritingCommunity https://t.co/LeuyYqVGA0',
     'Name this game. Wrong answers only. https://t.co/bEnEeoDN8x',
     'Add a caption. Any caption. #CaptionThis https://t.co/rrhFik8bka',
     'Im confused. Did Vladimir Putin change his name to God? https://t.co/z7MAo5h1QS',
     'Maybe its a message God did NOT support Trump in 2016, and that supporting him has always been stupid sinful. https://t.co/sl4laXrJnw',
     'Name this film. Wrong answers only. https://t.co/Sfit4JeSnS',
     'Add a caption. Any caption. #CaptionThis #WritingCommunity https://t.co/ifKJ1q0W7J',
     'Goddamnit. How did I forget Alien?!',
     'Was it this one? https://t.co/2LrnbdlT3a',
     'I remember that Phantom Menace trailer dropping. I always thought this Episode II one was solid trailer as well. https://t.co/DRnYFcxftA',
     'What are the top 5 movie trailers youve ever seen? Mine are:\n1. The Two Towers  (Trailer 2)\n2. Inception  (Trailer 2)\n3. The Fellowship of the Ring  (Trailer 2)\n4. Man of Steel  (Teaser 1)\n5. X-Men: Days of Future Past  (Teaser 1) https://t.co/bVYHeCQjqq',
     'Two months later: https://t.co/1rILAre1zX https://t.co/dnyw2egdW9',
     '"What does Trump look like? V-shaped. Fit. And hes definitely packing meat down there."\n\n...said no one ever. https://t.co/HZU2hyxebx',
     'Name this man. Wrong answers only. https://t.co/c4VamPT5sN',
     'At least Sen. Kennedy gave us a photograph that perfectly captures his shortcomings. https://t.co/aPHc96KKaf',
     'Another Confederate monument comes crashing down. https://t.co/Cw34Fi7Bfc',
     'Add a caption. Any caption. #CaptionThis #WritingCommnunity https://t.co/PuLcKA3y1P',
     'I cant believe this took so long. https://t.co/8TnKJcwO1K',
     'Name this man. Wrong answers only. https://t.co/zLY08WR74c',
     'Add a caption to this dame. (Dame Helen Mirren, to be specific.) #CaptionThis #WritingCommunity https://t.co/KgBgnmM6VU',
     'They can start with this thing, which I can vouch does not make Albany any safer. https://t.co/WErBUN05Xd',
     'There are good cops out there. Keep them funded. The bad ones are literally choking people to death. Theyre killers. Dislodge them. https://t.co/w8LzEnMbPW',
     'Name this man. Wrong answers only. https://t.co/BiNPZYNBm5',
     'Disgusting. https://t.co/zdcmSlnZon',
     'Add a caption. Any caption. #CaptionThis #WritingCommunity https://t.co/iZY7Iioxs3',
     '"Donald Trump and the Republicans are willing to sacrifice the lives of Americans to the coronavirus in order to save capitalism â\x80\x94 and of course Trumps reelection chances. This is grotesquely evil behavior." https://t.co/hn6xJeUY0S',
     'This sounds completely legal. https://t.co/5rRslYGTXg',
     'Mood: https://t.co/WUdd9ObgZ7',
     'Sorry. I didnt know. It was just a link a friend sent me this morning.',
     'Add a caption. Any caption.  #CaptionThis #WritingCommunity https://t.co/6YAUs9wBEl',
     'Ugh. I see your point. I wont share them again.',
     'lol. Point taken. https://t.co/HPP0hEXkhI',
     'Cmon, please do this... https://t.co/BWsRDfMqFZ',
     'Great find! Thanks!',
     'The 1993 PC game Police Quest IV billed itself as written by former LA Police Chief Daryl F. Gates. I always thought the game was kinda racist. Watching this now, I realize it was hella racist. \n\nhttps://t.co/uqWlMxknxY',
     'BREAKING NEWS: Trump rally canceled due to bull-shitstorm. https://t.co/PtFwwZLvqT',
     'I cant believe this is a real photo.  #CaptionThis #WritingCommunity https://t.co/2nMl5ta6tA',
     'Fox News has racist writers?  #TuckerCarlson  #WritingCommunity https://t.co/oey2P4ov1d https://t.co/LKRvYCvRmj',
     'Dear #WritingCommunity.\n\nPlease check this out! (You might end up writing for these folks!) https://t.co/xaPJY889le',
     'Bumblebee Man did it better. https://t.co/AZUIqwCM0O',
     'Add a caption. Any caption. #CaptionThis #WritingCommunity https://t.co/Q2iAL2yjhL',
     'Check your voting status. https://t.co/sN7HtMdwzQ',
     'It would suck if we actually mined this thing and its like: "Psyche! Its fools gold!" https://t.co/PB5OkNcfl8',
     'What a gift. What a lovely gift! https://t.co/nnowWGQFEV',
     'I dont know! I havent seen Parasite yet. #NoSpoilers',
     'I have a method that would have predicted every presidential election since 1976. I call it: "Did the film that won the Oscar for Best Picture that year have a happy ending?" If it did, the incumbent party wins.\n\nI like it better than this one.\n\n https://t.co/M0Gvkeuamn',
     'Name this man. Wrong answers only. https://t.co/8reqdYatNP',
     'Add a caption. Any caption. #CaptionThis #WritingCommunity https://t.co/zNqBKWsQxw',
     'Thats my biggest complaint as well.',
     'Its better than the books. https://t.co/pufvkfY6v7',
     '"Why does everyone think this is so funny?" - Christians Against Womens Suffrage after posting this. ð\x9f\x91\x87 https://t.co/zufgB4B5xo',
     'Full disclosure: this discussion is two months old. I just thought Id post it here. https://t.co/eBzXnIxYr6',
     'If Trump is forcing the US to be too sick to vote him out of office, should that be considered genocide? Your thoughts...  #WednesdayWisdom https://t.co/eA6qVQYQIl',
     'The discussion is two months old and has been raised by wiser people. I just thought Id ask. https://t.co/eBzXnIxYr6',
     'Add a caption. Any caption. #CaptionThis https://t.co/BJuSnVaFJY',
     'Sounds like Fox News is "mistakenly" full of shit. https://t.co/qvQ8fmCMgQ',
     'The United Daughters of the Confederacy is, and always was, a hate group. https://t.co/JRqkkx0OaY',
     'Since #ThePrestige is trending, I thought Id share this little nod I always liked. https://t.co/E1NcFQCthz',
     'Since Donkey Kong County is trending, I thought Id share this sick commercial. https://t.co/GPJLXFwoJh',
     'Name this show. Wrong answers only. https://t.co/XOFXpxbCvG',
     'Add a caption to this picture.  #CaptionThis #WritingCommunity https://t.co/t1wCoI5MqW',
     'If Ayn Rand were still alive, shed remain too hypocritical to die of shame. https://t.co/orDHe6YVhn',
     'Cmon. The worst place in NYC is obviously Trump Tower. https://t.co/XscMyPMxo4 https://t.co/rBF3aR2651',
     'I read this book in a single afternoon and enjoyed it. If youd trade the summer of 2020 with any summer from your past, you might like it as well. 4.25 stars. Full review on https://t.co/0xXphlBlXk #WritingCommunity https://t.co/1TnvHzJMNt',
     'Add a caption. Any caption. #CaptionThis #WritingCommunity https://t.co/38JfE15Yyh',
     'Yup. Its still 2020. https://t.co/CezD96Hfxj',
     'Please let me know if your experience is similar.',
     'At beast hes a villager from Ocarina of Time.',
     'I just watched Red Riding Hood, which played like it really wanted to be a Castlevania movie, then a Zelda movie, then finally just a movie. https://t.co/hpqBTze2Fo',
     'Name this film. Wrong answers only. https://t.co/71j4qv0RE6',
     'I just read this for the second time. Its so good. Please get yourself a copy!  #WritingCommunity https://t.co/7tcfHz1GZk',
     'Add a caption. Any caption.  #CaptionThis #WritingCommunity https://t.co/mA7dhofRDy',
     'I couldnt be happier for you. ð\x9f\x98\x84 Congrats! https://t.co/HmtZrHHex5',
     'Name this man. Wrong answers only.  #Happy4thofJuly #Happy4th https://t.co/Wt9m3oTQw2',
     'According to the Pew Research Center and the U.S. Census Bureau, Gen Z begins at 97. https://t.co/KNFWbZJZA7',
     'FACT CHECK: If youre 24 years old, youre not a Zoomer. Youre Millennial. https://t.co/xiy3nHRyTC',
     'Yes it is.',
     'People more historically significant than Billy Graham and Antonin Scalia:\n - Benedict Arnold\n - John Wilkes Booth\n - The shark from Jaws\n\nAm I missing any?  #WritingCommunity https://t.co/0lXUvF6S89 https://t.co/y4LXKsospG',
     'Really? These guys? ð\x9f¤¦â\x80\x8dâ\x99\x82ï¸\x8f https://t.co/EkF3oFJ5mP',
     'Name this film. Wrong answers only. https://t.co/LaJe0z4vmV',
     'Such citizenry sickens me. https://t.co/NzGRRtNGTq',
     'Im a huge fan and admirer of Guillermo del Toros filmography, but if forced to choose, I would have voted Get Out for Best Picture that year.',
     'offered a much better critique Nolans Dunkirk than I can fit into 280 characters, so I defer to them: https://t.co/ckpmhhDRjn',
     'Im guessing only white people fought in Dunkirk in the Tarantino movie universe? (P.S. This five-minute clip from Atonement was a better film than Dunkirk.) https://t.co/VUwB735HVp https://t.co/SKc4ogZsJ1',
     'Add a caption. Any caption. #CaptionThis https://t.co/KAgwdgOMsj',
     'Please read this thread for a glimpse into what its like to be a writer amidst COVID lockdowns.  #WritingCommunity https://t.co/hNdhjL9PK3',
     '...again? https://t.co/65rBTLvMhI',
     'If you stayed home, if you played it safe, if you chose to err on the side of caution, thank you. https://t.co/c3gyLQw6tG',
     'Add a caption. #CaptionThis #WritingCommunity https://t.co/552ftZEEPG',
     'Impressive!',
     'Name this vehicle. Wrong answers only. https://t.co/JOUHqkQtJ7',
     'There comes a point when you have more to lose from your possessions than you have to gain from them.  https://t.co/yQgAYReALl #WednesdayWisdom',
     'This could work. https://t.co/fibW7qgndH',
     'Add a caption. Any caption. #WritingCommunity #CaptionThis https://t.co/3ihsy1NMPd',
     'It sure is, and we hope you and Putin lose.',
     'BREAKING NEWS: Oklahomans would rather live with Obamacare than die under Trump. https://t.co/dS1dK7HxlX',
     'Name this show. Wrong answers only. https://t.co/sAKtDinD7V',
     'Not an illustration. Just busty.',
     'Upstate New York is strange. #CaptionThis #WritingCommunity https://t.co/xMSe9sLXKz',
     'Please read this moving essay from a Shakespeare troupe that cancelled their Summer season due to COVID-19. https://t.co/lGrrJcQ7Lm https://t.co/ceOsptIVeh',
     'For those interested, heres what Mitch McConnell looks like with his mask off. https://t.co/4QYZfIhclV',
     'Name this film. Wrong answers only. https://t.co/lfwAJ5SVJR',
     'Add a caption. Any caption.  #CaptionThis #WritingCommunity https://t.co/wHDUr8S5aX',
     'They gave us this guy. https://t.co/JXaih6eU3j https://t.co/ljAf6v8l0y',
     'Name this film. Wrong answers only. https://t.co/FNF4PLb8bh',
     'Two words: Trump + racism. https://t.co/cBw99yrUE1',
     'Add a caption. Any caption.  #CaptionThis #WritingCommunity https://t.co/OeqNOZpIuO',
     'This should make a fine campaign ad. https://t.co/5YgNAkKjou',
     'I like it already.',
     'Name this film. Wrong answers only. https://t.co/tsHiEZsjtG',
     'Your thoughts? (Be honest.)  #WritingCommunity https://t.co/X17mW79aUi',
     'Add a caption. Any caption.  #CaptionThis #WritingCommunity https://t.co/qMJvuptCLG',
     'Expect Republicans to acknowledge this sometime never. https://t.co/EbxBgudc2D',
     'Name this man and his occupation. Wrong answers only. https://t.co/5XRTKHDxDU',
     '"Itâ\x80\x99s always been how comfortable are you with one party over the other. Bernie is in no way a Socialist. Trust me. Germany has free public education and free healthcare. Itâ\x80\x99s hardly considered a Socialist country. But in the US, thatâ\x80\x99s taboo. ð\x9f¤·ð\x9f\x8f» â\x80\x8dSo, super hypocritical." (8/8)',
     '"The point is, it is only political rhetor[ic] to label Socialism to a particular party, mostly as scare tactics. US never ceased to throw out financial assistance in large amounts under any of the two parties." (7/8)',
     '"Well, just last year alone, US farmers received billions in bailout as a direct result of our trade war with China. Most recently, the stimulus check to [individuals] originated from GOP [Senators]." (6/8)',
     '"In addition, many average Americans understand Socialism as govt financial assistance to sectors of labor or manufacture like farming, auto industry etc,. and it has [been] heavily linked to the Dems [as] Obamacare." (5/8)',
     '"Also, thereâ\x80\x99s no [free] press or free speech and â\x80\x9cmonitoredâ\x80\x9d freedom of religion (whatever the fuck that means) because they want to protect the 1 party rule. So yeah, fuck that all the way." (4/8)',
     '"I grew up around the time food rationing was still around in Vietnam. Thats Socialism. I didnâ\x80\x99t remember the exact details since I was still little, but my grandparents tell me itâ\x80\x99s like a couple hundred grams of meat, oz of milk &amp; few cups of [rice] per person everyday." (3/9)',
     '"There are currently 5 countries in the world following true model Socialism: China, Vietnam, Cuba, Lao and North Korea. They have lots of glaring common denominators, but 1 standout most: single political party. To start a 3rd party as Socialist is itself a contradiction." (2/8)',
     'I asked a friend of mine from Vietnam, a doctor in Florida, what he thought about Socialism in the USâ\x80\x94specifically, Socialist groups like that supported Sen. Sanders but are campaigning against Joe Biden. This was his response, and he asked me to share it. (1/8)',
     'When Sen. Cotton says "statehood," does he mean Union or Confederate? https://t.co/szVh40RmOJ',
     'Add a caption. Any caption.  #CaptionThis #WritingCommunity https://t.co/KyaOkBim4U',
     'Great. Now Im picturing Boltons naked neoconservatism in Eden.  https://t.co/5rSNNL2WhX',
     'Name this film. Wrong answers only. https://t.co/ZwzG9Z34jo',
     'You should!',
     'For Trump, these polls show he has a remarkably stable base and high ceiling, but hell need more than that to win. Trumps current strategy is a losing oneâ\x80\x94unless he changes tactics or the GOP ups its voter suppression to 11. A smart opponen',
     'These polls show Biden has a much higher ceiling than Sec. Clinton ever did, which is a plus since all he needs is to improve her #s by 1% in WI, MI, &amp; PA to win. They also show down-ballot Dems across the country could benefit from his co',
     'Im a former presidential campaign staffer. At this point in a campaign, polls are more of a diagnostic tool than an indicator of what to expect in November, but they also play key roles with respect to spending and summer strategies.\n\nIn short',
     'According to the latest FOX News survey:\n\nTEXAS\nBiden: 45%\nTrump: 44%\n\nFLORIDA\nBiden: 49%\nTrump: 40%\n\nGEORGIA\nBiden: 47%\nTrump: 45%\n\nNORTH CAROLINA\nBiden: 47%\nTrump: 45%\n\nhttps://t.co/ap7NRBioOB',
     'Its tough to live with something a hundred thousand Americans already died from. https://t.co/Zk1tKZrF3u',
     'Please ask this man anything. Like Wolverine, hes the best at what he does.\n #WritingCommunity https://t.co/yJwnXlO4tq',
     'Add a caption. Any caption.  #CaptionThis #WritingCommunity https://t.co/YiSnUtxaWe',
     'If youre curious how Republicans will try to steal the presidency yet again, itll be by denying these voters ballots. If that happens, the United States should rightfully be called a failed republic and its leadership a dictatorship. https://t.co/vjO5o25Uhl',
     'In honor of International Fairy Day &amp; birthday (Happy Birthday!!!), whos your favorite fairy in film history? Mine is Oona from Legend (1985), the closest film we have to a Legend of Zelda movie.  #WritingCommunity https://t.co/Fg0jJSyjVu',
     'Name this film. Wrong answers only. https://t.co/sxYNIdJ9yN',
     'TRANSLATION: Top GOP senator STILL has no problem with Trumps racism, fascism, or criminal behavior. https://t.co/3JMBp3BwYe',
     'Conservatives got their 200 judges. All it cost was 30 million jobs, 120k lives, and the US Constitution. https://t.co/ORDEJhQqj6',
     'Add a caption. Any caption.  #CaptionThis #WritingCommunity https://t.co/f7qJpJI6xv',
     'Of course! https://t.co/Vk9eeEBV0G',
     'Congrats! Id love to read it too!',
     'Name this film. Wrong answers only. https://t.co/eUUBri32OE',
     'Since Burtons #Batman is trending, can we talk about Bob the Goon? This guy:\n â\x80¢ Takes Leibovitz-level photos\n â\x80¢ Can stage a parade or mime performance in minutes\n â\x80¢ Inspired Kevin Smiths "Silent Bob"\n â\x80¢ Died as a punchline for his boss\n\nDude might the greatest henchmen ever https://t.co/wQ7nIOGqmL',
     'Oh yes. I had that toy.',
     'Damn. They grow so fast! https://t.co/4Ow3GnLbnh  #StudentsForTrump',
     'This is what Jim Crow laws look like today.  #KYPrimary https://t.co/1gCNAByS0N',
     'How does this guy still have a job? https://t.co/JgIYQ5AsWm',
     'Was it this one? Cause thats what I had.  #Batman https://t.co/or42sb3j1z',
     'On June 23, 1989, Tim Burtons BATMAN hit theaters. I was 5 y.o. at the time, and it was the first blockbuster I saw. After that, Batman was everywhere: t-shirts, parties, cereal bowls. It literally canvased my childhood. Whats your fav. memory of BATMAN? https://t.co/IyPTJhykUC',
     '#GiveTwoFucks should be a hashtag. Its something we could all benefit from. https://t.co/iWS62SpiS9',
     'Everyone on the Trump campaign right now. https://t.co/S8NfkWJQNZ https://t.co/RYD9BO0xiu',
     'Impressive!',
     'Nah, separate. (A fascinating idea, though!)',
     'Today Im writing about:\n â\x80¢ La Noche Triste\n â\x80¢ Columbus First Voyage\n â\x80¢ Andrew Jackson and the Cherokees\n â\x80¢ H. P. Lovecraft\n\nYou?\n\n#WritingCommunity https://t.co/SQcpnts9ZD',
     'Add a caption. Any caption. #CaptionThis #WritingCommunity https://t.co/H35kEGWQVv',
     'If having a Black husband excused ones racism, Donald Trump would brag about having several.  #Karen https://t.co/izV8MdQzpB',
     'Wheres B. J. Blazkowicz when you need him? https://t.co/WdGmGvKnX0',
     'Oh, how the tables have turned.  #McSally #LiberalHack https://t.co/6dAs8Kczn9 https://t.co/G8UYwYiozH',
     'After reading this, Im 100% down with a Star Wars Infinities that explores what would have happened if Qui-Gon defeated Maul and raised Anakin as his apprentice. https://t.co/4joEKy8JQg',
     'To you, https://t.co/FUDcKkThK5  #MacTrump',
     'My love and I watched Charlie and the Chocolate Factory last week. Our diagnosis: if Burton directed it a decade earlier w/ Michael Keaton, they would have nailed it. https://t.co/mysSRZPF52',
     'Did you seriously amend your statement because it was not racist enough?',
     'Add a caption. Any caption. #CaptionThis #WritingCommunity https://t.co/itKZRMfh3W',
     'Last week, it was statues of Confederates, Columbus, and Frank Rizzo, and rightly so. \n\nThree days ago, it was Grant, which I disagreed with but understood.\n\nIf anyone goes after #Hamilton, Im immigrating back to Italy to see the David before some tourist knocks it down. https://t.co/lwJbIDpaRv',
     'I imagine its comparable to appreciation for certain aspects of Alexander Hamilton.',
     'Haha! I love it!',
     'Please tell me youve heard this, because its brilliant. https://t.co/Ul2TEqDgrM #WritingCommunity',
     'Name this man. Wrong answers only. https://t.co/LsaE2tzHcD',
     'Ever since his Death Star tweet, I picture as that nervous guy from the beginning of Return of the Jedi. https://t.co/KZ7J6hmJ5f https://t.co/E0u14ojzQn',
     'Happy Fathers Day, everybody! https://t.co/4yoDtZIG5G',
     'This might surprise some people, but sometimes I tweet a picture just because I like it. Take this one. I think this is a good picture of Mitt Romney. I dont know why, but it looks like how I imagine the POTUS would look in The Dark Knight. https://t.co/bWDjaWTpQF',
     'Looks like Trumps Death Star is not operational as planned. https://t.co/1rILAre1zX https://t.co/41t4NymCZL',
     'I also deliberately follow and unfollow cybertrolls for months at a time so they can be better identified and removed. If this bothers you, I apologize. https://t.co/i3ZDXARkQs',
     'Name this film. Wrong answers only. https://t.co/vQO6RW9Wey',
     'While Im down for every statue of Confederates, Columbus, quite a few Founders being relocated to museums (graffiti included), can we please stop shaming people who dedicate their lives to historiography? For all we know, the guy on the right is working on his PhD. https://t.co/CfdpP7rS7F',
     'Oh, theres an entire world of difference, which is why I asked for clarification. (Which she gave and I am grateful for.)\n\nThat said, according to these replies, it looks like quite a few Americans want ERs statue tor',
     'I ask because, frankly, statues are an antiquated and questionably efficient form of public display, especially since theyre so easily torn down. If books, music, and film can reach more audiences with greater effect, should they be destroyed ',
     'Im not arguing. Really. I see your logic. Im just trying to understand its end result. "Amazing Grace" was written by a former slave trader. Should that song be barred? Frederick Douglass ultimately favored black suffrage over womens suffrag',
     'By that logic, we should tear down statues of Eleonor Roosevelt because her husband interned Japanese-Americans.',
     'Ian Holm was an artist who accented every film he made the way frames accent a masterpiece or wine accents a perfect dish. He will be missed.  #RIPIanHolm https://t.co/tHLBxEresD',
     'Name this man. Wrong answers only. https://t.co/DdXF16X9sK',
     'Name this film. Wrong answers only. https://t.co/8m8u0Gge74',
     'Add a caption. Any caption. #CaptionThis #WritingCommunity https://t.co/6mGDGxoXk7',
     'I love nature. https://t.co/tQ7scQbDVP',
     'Name this film. Wrong answers only. https://t.co/JvLwSGLVPO',
     'Provide a fascinating theory about these tiny holes. #WritingCommunity #CaptionThis https://t.co/VGrLdnB6S7',
     'The parents gravestone saved their childrens!',
     'Pictured: damn good parenting. https://t.co/zQRS2w8JJu',
     'Honestly, I cant think of a better monument to the Confederacy than an empty pedestal. If there are no objections, I suggest we keep a few of them that way. https://t.co/8vvP8h5QvS',
     'This cant go wrong. Right?? https://t.co/XxN6tDRx6w',
     'Name this film. Wrong answers only. https://t.co/mY1A2b84uQ',
     'Add a caption. Any caption. #CaptionThis #WritingCommunity https://t.co/zRs2U4itI2',
     'Thanks!',
     'Thank you!',
     'I dont do this often, but the replies to this are brilliant. Thanks! #WritingCommunity https://t.co/DT2Phevnun',
     'Netflix right now. https://t.co/6ysc4fCAeo https://t.co/gAykz6Ie30',
     'Add a caption to this kitty. #CaptionThis #WritingCommunity https://t.co/BRVogUPOCJ',
     'All we need now is a playable version of My Dinner With Andre. https://t.co/K04EwOOfQv',
     'I have an unpopular opinion that anyone this bigoted should have no place in our government and be removed, particularly from our courts. https://t.co/0bezMIWV60 #PrideMonth',
     'My honest-to-God suggestion: replace every statue of Christopher Columbus in Philadelphia with Danny DeVito. https://t.co/CQY4EqiUWt',
     'For those curious, a "silent majority" never existed. It was just a loud minority. Read about it here: https://t.co/L1KMMGwQSa https://t.co/ZzPisImlkT',
     'YOUR RE-ELECTION IS LOOKING WORSE THAN EVER!!!',
     'Name this thing. Wrong answers only. https://t.co/sooGQPjag0',
     'Add a caption to this classic. #CaptionThis #WritingCommunity https://t.co/xIbzADYG2P',
     'Team Avatar was antifa. Thats why theyre heroes. https://t.co/YcIGAijIbc',
     'Watch this video. Share this message. https://t.co/6HcnAe3n3S  #BLM #JayPharoah',
     'Add a caption. Any caption. #CaptionThis #WritingCommunity https://t.co/gA1HLIJ7vU',
     'You know whatd be pathetic? If this guy and his descendants spend the next 155 years erecting statues in his honor, insisting he was the greatest NASCAR driver that ever lived. https://t.co/GdQyqpjTla  #ConfederateFlag',
     'Lets settle this: Whats the best episode of Avatar: The Last Airbender? Explain in detail.  #WritingCommunity https://t.co/GftnRmxNJA',
     'Heartless. Thoughtless. Hateful. https://t.co/lRRLiecmSN',
     'Unpopular opinion: I always thought this broadcast-friendly scene from Ghostbusters was funnier than the original. https://t.co/s13U8dTBDe',
     'Add a caption. Any caption. #CaptionThis #WritingCommunity https://t.co/mlCCGZWPpX',
     'Theres something wrong with my keyboard. I keep spelling Tucker with an "F" and "KKK." https://t.co/lUVhRM4HpV',
     'Name this film. Wrong answers only. https://t.co/cmDwaW58zt',
     'Hey, if it aint broke... https://t.co/fpcmj9a2tD',
     'Im teaching a FREE online class on the Civil War in about an hour. Please join in! https://t.co/Otu8Jqmo4z https://t.co/dZz2XLzQCe',
     'Thank you!',
     'Expect to call Rupert Murdoch a closet Democrat any second now. https://t.co/9KfLUzJBwy',
     'Add a caption. Any caption. #CaptionThis #WritingCommunity https://t.co/9ssHe8lV98',
     'Since someone asked, the following books and documentary provide excellent histories on Reconstruction from different angles. I recommend them. (Also, please support your local bookstore by ordering from them.) #WritingCommunity https://t.co/73tBJpQfb2 https://t.co/CM2YazaQ91',
     'I will be teaching a FREE online class on Civil War military history this Thursday, 7 PM (EST), through If youd like to join it, click here for the details! https://t.co/Otu8Jqmo4z  #WritingCommunity https://t.co/8UNnjpIPPc',
     'Name this film. Wrong answers only. https://t.co/1bqGKCmG7J',
     'Joshua Lawrence Chamberlain is more worthy of commemoration than every Confederate that ever lived. https://t.co/9OuMYteWvW https://t.co/0ryk9znnRV',
     'Add a caption. Any caption. #CaptionThis #WritingCommunity https://t.co/qNwXEKqsba',
     'Gone With the Wind is a cinematic masterpiece and an encyclopedia of Hollywood history, both good and bad. But cmon, the film openly weeps nostalgia for some of the most abhorrent Americans that ever lived. The films message is racist, hateful trash. https://t.co/9OOtO5qUUS',
     'Its possible to appreciate the creative talentsâ\x80\x94sometimes geniusâ\x80\x94in propaganda w/o subscribing to its intended message, but most Gone With the Wind fans arent like that. They worship the film with nostalgic eyes despite it being a high-end production of Confederate pornography.',
     'FACT CHECK: Gone with the Wind is the most successful work of Confederate propaganda in history. https://t.co/J1kt5jcNJ5',
     'I feel like most people protesting the removal of Gone with the Wind from HBO already own a copy on Blu-ray, DVD, and VHS. https://t.co/y3IuPfiYQO',
     'Thank. You. https://t.co/9QZKDDx9pY',
     'Add a caption. Any caption. #CaptionThis #WritingCommunity https://t.co/hOBm7ffuY8',
     'A damn fine book as relevant today as it was before COVID-19. I recommend it. Full 5-star review on https://t.co/3kTc58KVTf #WritingCommunity https://t.co/T7Rc0fEmGK',
     'I asked my wife to invent an ending for Clue where Mrs. White killed Mr. Boddy. I think she nailed it: "Mr. Boddy was her husband who disappeared."  #WritingCommunity https://t.co/0pSc9cOeEW',
     'Police brutality doesnt begin and end with bad cops. It includes bad police unions, sherrifs, judges, district attorneys, prisons, and elected officials at the state, local, and federal level. This is why you MUST vote and organize beyond the hashtag. https://t.co/ZGHaeyV14v',
     'I support the most extensive police AND prison reforms in US history, but only if theyll protect good cops like Krystle Smith, FLPD (below) and former BPD Cariol Horne. They are the whistleblowers our country needs right now, and they should be protected. https://t.co/ISTxSguw0L',
     'Add a caption. Any caption. #CaptionThis #WritingCommunity https://t.co/4VI9ViLIOJ',
     'Well Ill be damned. Willy Wonka took place in Chicago, Illinois. https://t.co/Pith83dtbO',
     'FACT CHECK: Barrs an idiot. https://t.co/MZjqNbbOw4',
     'Just wear a mask, already.  https://t.co/xiXzItNz1L',
     'Hi, and welcome!',
     'In lighter news, The Goonies came out 35 years ago today. https://t.co/dTR48HVM0K',
     'If you were Indian under Churchill, you were already fckd. (By Winston Churchill.) https://t.co/iJ8tcpxwuQ',
     'Australian Man is so much better than Florida Man. https://t.co/phmCC5hMbV',
     'No one should ever have to fear saying, speaking, or thinking these words. #DDay76 https://t.co/KIzGFFOlHf',
     'If you think fictitious ELVES are more deserving of equal rights and compassion than human beings, your moral compass is as broken as a Horcrux.\n\nThats all I have to say.  #PrideMonth',
     'Add a caption. Any caption. #CaptionThis #WritingCommunity https://t.co/ckM5ANWehw',
     'The Confederate battle flag was flown by enemies/traitors to the US military, including those who murdered POWs and US Marines.\n\nTake it down and tear it up. https://t.co/mHd5pvcsHT',
     '76 years ago, the biggest antifa event in history began.  #DDay76 #DDay https://t.co/Ix7ryM6gtF',
     'HBOs Watchmen is up there with Children of Men as one of the most relevant works of fiction right now. https://t.co/o5sxEl6G2z',
     'Thanks so much!',
     'Name this show. Wrong answers only. https://t.co/noBX0AbX2T',
     'I am so sorry.',
     'If this is what our law enforcers look like, we should consider law enforcement a failed experiment and start from scratch. https://t.co/EAAveLb1Fi',
     'Add a caption. Any caption. #CaptionThis #WritingCommunity https://t.co/SwdGYGebLI',
     'Ladies and gentlemen, I give you... THE THIRD AMENDMENT! https://t.co/C7tV1odmXs https://t.co/Jjh8V06hUU',
     'Looks like Trumps ego has finally offended Zeus. https://t.co/SoLyT9xG62',
     'I wonder if theyll arrest the Statue of Liberty for loitering. https://t.co/WVkgUBaCc5',
     'Name this film. Wrong answers only. https://t.co/bdq0v1OQCp',
     'It could be worse. He could have tweeted this. https://t.co/aEFupdBQda',
     'All that force against one man? Hes stronger than theyll ever be. https://t.co/FRCbqFTjEU',
     'I just learned my cousin and his crew helped save three kids and a dog! I could not be prouder.  #DayMade https://t.co/6v9GNlHWTn',
     'Your tax-dollars at work. https://t.co/fTE8upYIoo',
     'Add a caption. Any caption. #CaptionThis #WritingCommunity https://t.co/yqdelCcNrW',
     'I hate to revisit this, but Sen. Murkowski isnt struggling. Parents are struggling. Teachers are struggling. POC have always been struggling. Our economys a mess. Our Republic is suffocating.\n\nSen. Murkowski, youre just what youve always been: wrong.\n\nhttps://t.co/t1ZfqtEpQp',
     '"My Struggle," by Lisa Murkowski doesnt sound like a smart PR move. https://t.co/nvFruii1bc',
     'This is coming from a man who just dyed his hair Trumps color. https://t.co/IYI4rzeu7d',
     'Tom Cotton committed a felony in 2015, so who gives a crap what he thinks? #ThrowbackThursday https://t.co/GhSQdBvxkw',
     'This man is a living, breathing Confederate statue. https://t.co/1Xuxv7IiO9',
     'FACT CHECK: Nope! https://t.co/cAM3hLCbkS',
     'Not since Andrew Johnson has a president been so rejected by the military. https://t.co/dvAhgHdW0z',
     'Add a caption. Any caption. #CaptionThis #WritingCommunity https://t.co/3y0mkGtiEW',
     'What a buried headline... "The [FBI] report did warn that individuals from a far-right social media group had â\x80\x9ccalled for far-right provocateurs to attack federal agents, use automatic weapons against protesters.â\x80\x9d https://t.co/qHSYpuLjU9',
     'Trumps use of tear-gas to dispel a peaceful protest and even clergy from a church is the stuff of dictatorships. It is anti-American and inexcusable. The best Senate Republicans could do was raise their hands in indecision. VOTE THEM OUT. https://t.co/I25u7XDfEl',
     'Name this film. Wrong answers only. https://t.co/XycIQw1J6o',
     'Id celebrate this if it werent 18 years too late. Shame on you, https://t.co/CSazv5HJPG',
     'Keep the peace. Maintain the movement. Report and record crimes when they happen. And please be safe. https://t.co/FykARiCkcr',
     'This is tyranny. Also, this is America. https://t.co/uK1L1rpKvD',
     'Pictured: progress. https://t.co/qQq0IWTFao',
     'Add a caption. Any caption. #CaptionThis #WritingCommunity https://t.co/TXhYxZIMa7',
     'Hell, they called him that even when he didnt do it. https://t.co/FsznBkrgXE',
     'This is absolutely horrifying. https://t.co/9bDrvoNR7O #TheBronx',
     'I hope you have a witty response for this, because this isnt funny. https://t.co/ry8wl1yBoD',
     '"Then the Lord said to Moses, â\x80\x9cGo down, because your people, whom you brought up out of Egypt, have become corrupt. They have been quick to turn away from what I commanded them and have made themselves an idol cast in the shape of a calfâ\x80\x9d  - Exodus 34:7 https://t.co/cVfRE4QJvg',
     'Now seems like a good time to share the Bibles Trumpiest passages:\n\n"False messiahs and false prophets will appear and perform great signs and wonders to deceive, if possible, even the elect." -  Matthew 24:24 https://t.co/nQugXjBFE3',
     'Just a reminder that, in addition to everything, the oceans are still rising, COVID-19 is still rampant, and children still need an education. https://t.co/qasQZ6mg2Y',
     'Add a caption. Any caption. #CaptionThis #WritingCommunity https://t.co/ST0fVCv0Rf',
     'Official autopsy: Not a homicide.\nIndependent autopsy: HOMICIDE.\n\nSee the problem here?\n\nhttps://t.co/7og4txbeWD',
     'It was right after Columbus Day, so you can imagine how warmly that was received! When later asked how I could believe such things about Columbus, I had to point at pages from his journal and ask "have any of you even read this?" Not',
     'Speaking as an Italian-American, I can assure you the groups defending such monuments have such an unhealthy idolatry that Ive heard them defend Confederate statues as well. I once truth-bombed them at a lecture, explaining "these p',
     'No issue here, https://t.co/HDWMZFgcLa',
     'Not pictured: "winning."  #BunkerTrump https://t.co/zZl9y085ZQ',
     'Remember four years ago, when this didnt happen? https://t.co/RRFd9gHNFR',
     'Its a tie. Theyre all perfect. https://t.co/r1wOKHwmXb',
     'Name this film. Wrong answers only. https://t.co/MZ9ysDIey7',
     'Its not an assertion. Read the book. I was skeptical at first, but the quantitative evidence in it persuaded me.',
     'This seems relevant. https://t.co/2uK1aTaKAe',
     'Yeah. And the KKK is just a political meeting. https://t.co/xvvQfUW9MO https://t.co/BsjH4Z4vH6',
     'Add a caption. Any caption. #CaptionThis https://t.co/oLpiEHRRqO',
     'Lax law enforcement standards, oversight, and punishment have provided the perfect habitat for white supremacy to thrive. If youre a Klansman or Neo-Nazi, theres literally no better job in America for you and your movement. (Well, maybe president.) https://t.co/j6rzdQZgEh',
     'I like horses too, but if 100 black horses were shot and killed by law enforcers every year, that shit would END. And thats the problem. https://t.co/7Nbfu4yJBv',
     'Just a reminder, this is what Albany NY looks like in peacetime. https://t.co/WErBUN05Xd',
     'Name this film. Wrong answers only. https://t.co/nXeQtVsaTb',
     'Name this film. Wrong answers only. https://t.co/o5KHx7N2h1',
     'Add a caption. Any caption.  #CaptionThis #WritingCommunity https://t.co/2OY7JGA8vO',
     'Thanks for clarifying that. I changed the picture.',
     'Has anyone ever seen a white, armed protester get arrested? Ever? https://t.co/Fw3rwDF4nb',
     'Im down!  #WritingCommunity https://t.co/GzchHzkYvx',
     'Thank you for writing it! It is most enjoyable, and I look forward to your future works.',
     'That was my first thought as well.',
     'Name this film. Wrong answers only. https://t.co/BseiwftR35',
     'Its rare to read a book that does not waste a single paragraph. The Lost History of Liberalism by is one of the most thorough yet accessible academic texts Ive ever read. A 5-star book. Full review at \nhttps://t.co/LQ0JOy2kjq #WritingCommunity https://t.co/mShTXiZQwz',
     'I see a lynching by a racist using the "thin blue line" as a noose. https://t.co/IECccraJK1',
     'Remember when Republicans called Obama fascist for wanting to cover pre-existing conditions? Theyre now fascists. https://t.co/bi2DLikTWI',
     'Add a caption to this handsome devil. #CaptionThis #WritingCommunity https://t.co/kTd96TRt5S',
     'Name this film. Wrong answers only. https://t.co/wyoHB8ZWzK',
     'This is what presidents used to look like. https://t.co/gX36fGsvD2',
     'The Texas Supreme Court just ruled the risk of COVID-19 does not qualify someone for an absentee ballot. Since they decided this vote remotely due to the risk of COVID-19, I suggest we consider their decision a self-own. https://t.co/I8DIUlflhp',
     'Me: "Are you a werewolf?"\nMy fiancÃ©e: "I AM that bitch."\n\nAnd thanks! From both of us, youre welcome! https://t.co/WCT52dQvJe',
     'I asked an expert. She said since "let em hang" is not an option for all woman, shed buy armfuls of whatever bras are on clearance. She added: "My breasts physically hurt just thinking about this poor werewolf. What if she works at a place wi',
     'Guilty as charged. https://t.co/YnfQEBVhRe',
     'Really, I couldnt be happier for you! I wish you the best of luck and hope it keeps coming.',
     'A thought: if youre a Sanders delegate who hosts town halls attacking Democratic candidates, you have no place in the 2020 DNC. (Also, how can you be a "PA Bernie" delegate when PA hasnt even had their primaries?) https://t.co/h34jSneTM6 https://t.co/N9wamlOrQE',
     'Add a caption to this specimen. #CaptionThis #WritingCommunity https://t.co/f2utzgY9ZX',
     'I bought Florsheims because of one line in Chinatown. We barely even see them in the film! (Great shoes, btw. Theyve lasted me over a decade.) https://t.co/sF4nfAIm2O',
     'Such good movies. Do you think Hoskins gave a better performance in Mona Lisa or Who Framed Roger Rabbit?',
     'I dont think its an elections fault Ghostbusters 3 didnt start like this. ð\x9f\x91\x87 https://t.co/ci6Y6KEats https://t.co/I6C9hY1rPZ',
     'Thanks, Scientist Man!',
     'Trumps ahead of Biden by just three points in... UTAH. https://t.co/xkeR4uSAZD',
     'Name this film. Wrong answers only. https://t.co/wfXok4JiqK',
     'FACT CHECK: what Trump considers "FREE SPEECH" https://t.co/dZgyaMy51r https://t.co/YYuNHp3Itx',
     'I agree. https://t.co/OAVyesLBEL',
     'Add a caption. Any caption. #CaptionThis #WritingCommunity https://t.co/bsEzebngAL',
     'From February to May 2020, the United States lost more lives to COVID-19 than to:\n â\x80¢ The Vietnam War\n â\x80¢ The American Revolution War\n â\x80¢ The Iraq War\n â\x80¢ The War in Afghanistan\n â\x80¢ Desert Storm\n â\x80¢ 9/11\n...put together. https://t.co/XUPtyDPOcs',
     'FACT CHECK: Wear a mask. https://t.co/otDoWDIfqY https://t.co/R9QDd4I2fj',
     'My thoughts remain the same.  #Grant https://t.co/iGAPVyUwoq',
     'Name this film. Wrong answers only. https://t.co/cQ1V94UrnE',
     'Im sure the Kremlin would be happy to host the RNC in Russia. https://t.co/ofCaSnOVso',
     'Biden laying a wreath for our fallen is more presidential than anything Trump has ever done. https://t.co/QU4CCYGqbp',
     'Lest we forget, nearly 90% of COVID-19 deaths couldve been avoided if the US took precautions a few weeks earlier. https://t.co/wotDOD99Sf #MemorialDay',
     'Add a caption. Any caption. #CaptionThis #WritingCommunity https://t.co/Yxt0qoR07i',
     'TimeSplitters 2 https://t.co/CCoz9LSYiM https://t.co/OEWfyxpkwv',
     'I thought Trump was "a very stable genius." ð\x9f\x98§ https://t.co/tc8BEdADui',
     'Name this film. Wrong answers only. https://t.co/OangZxaiVM',
     'I think its a fantastic way to illustrate the Libertarian Party has no ideas and will lose in November.  #ImWithHer',
     'Issues a blistering critique of Trump, has no problem with the racist, fascist policies he campaigned on and Russia aided.\n\nSmooth. https://t.co/CNrIg7f7sv',
     'Name this film. Wrong answers only. https://t.co/zYMmLXaR03',
     'I object to this. Medusa would be WAY more interested in Nagini than Voldemort. https://t.co/LO7aGkl6s0',
     'Describe the original Star Wars using just this poster for reference.  #WritingCommunity https://t.co/XijLI3RTrZ',
     'Some campaign ads write themselves. https://t.co/eBdjTaBBy1',
     'Add a caption. Any caption. #CaptionThis #WritingCommunity https://t.co/9bFA205zK0',
     'I always wondered why Yodas theme played during Leias escape from Cloud City in #EmpireStrikesBack. I now think I know: THIS was the moment Yoda glimpsed when Luke asked: "Will they die?" Thats right, Yoda is in this scene, FROM THE PAST! ð\x9f¤¯ https://t.co/vfdJ6bbo4X #Empire40th',
     'Name this film. Wrong answers only. https://t.co/rzVo95Hn9i',
     'Unpopular opinion: politics is not for everyone. It can lead to addictive behavior as destructive to ones friends and family as any drug. Im glad some "values" voters are breaking their Trump addiction, but their addiction to politics is just as harmful. https://t.co/7GwvIlKYAn',
     'Its nice to see people finally get Scott Pilgrim vs. The World. The film is a masterpiece, and is the Kurosawa of comedy. https://t.co/MUPfQHX81T',
     'Perfect. https://t.co/Q9ogjUl4eZ',
     'Add a caption to this beauty. #CaptionThis #WritingCommunity https://t.co/moVyQsBgOC',
     'God damn this year.  #EmpireStrikesBack #Empire40th https://t.co/mnG5SGZ1Sn',
     'Cool! It looks like a scene from The Matrix!',
     'Name this film. Wrong answers only. https://t.co/vz2pMNijoQ',
     'Trump has officially entered James Buchanan territory as our worst president ever. https://t.co/8HrrFiPqCm',
     'And a caption. Any caption. #CaptionThis #EmpireStrikesBack #Empire40th https://t.co/Xar6vWA1EZ',
     'About a year ago, I co-wrote a play called MacTrump. It re-imagined the first two years of the Trump administration as a Shakespeare play.\n\nIm not gonna lie: this particular passage creeps me out. Why? Because the "shiny pox" this character fears is called a "corona." https://t.co/UgbYZKHVSl',
     'Name this film. Wrong answers only. https://t.co/9mWb47XdJE',
     'Such a great movie. Ill be following!',
     'Shakespeare and the Folktale is the type of book I wish I had read years ago: a fascinating, entertaining tour of the Bards folkloric sources across the centuries and continents. Check it out! Full 5-star review on https://t.co/rn8pOB2xBT #WritingCommunity https://t.co/ws73ssty7n',
     'Add a caption. Any caption. #CaptionThis #WritingCommunity https://t.co/XhgFUYvzMH',
     'We need to save this for when Trumps no longer president. https://t.co/eHQTnisQ6i',
     'Goddamn. I thought Earthworm Jim was hermaphroditic all my life! (I think I learned it in Sega Visions.) I know its not the same as being trans or tolerant, but I cant think of any other superhero from my youth that identified with a gender while intersex.  #WritingCommunity https://t.co/Hib9TMh3mc',
     'If Earthworm Jim joins #SuperSmashBros, what should his final smash attack be? https://t.co/VaK6X2ORQA',
     'Name this film. Wrong answers only. https://t.co/lD62iMbxLC',
     'I guess "Thou shalt not bear false witness" doesnt apply to Christians who pay other people to lie. https://t.co/SL1inqAlm0',
     'Add a caption. Any caption.  #CaptionThis #WritingCommunity https://t.co/OslbfFrxc7',
     'I hope Trump continues to break tradition by being the first President to go to prison. https://t.co/LvcStvoOYX',
     'Majestic! ð\x9f\x96¤ Thanks so much!',
     'If Dantes Hell exists, Roger Ailes is deep in it. https://t.co/ACpVcIJNuM https://t.co/Vh0x49DvHH',
     'Name this book. Wrong answers only.  #WritingCommunity https://t.co/oYhH05x7i4',
     'Name this film. Wrong answers only. https://t.co/oizyDBLEBj',
     'My favorite athletes.\n\nBaseball: Babe Ruth\nBasketball: Scottie Pippen\nFootball: Joe Montana\nBoxing: Evander Holyfield\nSoccer: Gianluigi Buffon https://t.co/Jo9QeV6Kut',
     'Add a caption to this beauty. #CaptionThis #WritingCommunity https://t.co/44StxJwL70',
     'My God... A friend of a friend lost BOTH parents to COVID-19. ð\x9f\x98¥https://t.co/FsVjLKDyZC',
     'Name this film. Wrong answers only. https://t.co/amDH6awH4Z',
     '"Youre all losers. Obamagate." #TrumpCommencementSpeech https://t.co/eiCUXGP1YG',
     'My god...',
     'Add a caption. Any caption. #CaptionThis #WritingCommunity https://t.co/0KSTRmBshD',
     'We hold these truths to be self-evident: #ObamaWasBetterAtEverything https://t.co/yAVJ8Om2WI',
     'Name this film. Wrong answers only. https://t.co/kifKGriqpH',
     'Add a caption. Any caption. #CaptionThis #WritingCommunity https://t.co/2lr4NVGgRk',
     'Not gonna lie: my friends and family loved AotC when we saw it opening day. https://t.co/d9XF0KIrrX',
     'Looks llike Trump just lost a voter. https://t.co/4j6G8VGhcP',
     'Youre the best editor anyone could ask for! Stay safe and healthy, #ff',
     'Im getting really tired of this Old Testament "wrath of God" stuff. https://t.co/LU9L9MrdSb',
     'Hitler wouldve won WWII against these stooges. https://t.co/v9RudohsYu',
     'Add a caption. Any caption. #CaptionThis #WritingCommunity https://t.co/oZROqB279k',
     'Good news! The Constitution doesnt care. https://t.co/iJpPsrRqPA',
     'American exceptionalism has officially died of COVID-19. https://t.co/8iLw4v53Ti',
     'Im listening this right now. Please join in! (And ask a question.) #WritingCommunity https://t.co/kEnkOe1gQI',
     'Name this film. Wrong answers only. https://t.co/JmPuRLyeUZ',
     'The Michigan Legislature shouldnt close their doors. They should call in the National Guard. https://t.co/nUDcU3pchK',
     'What are the top 5 fictional books youd like to see printed? Mine:\n1. The Princess Bride, by S. Morgenstern\n2. Dr. Henry Jones, Sr.s Grail Diary\n3. Handbook For The Recently Deceased\n4. The Hitchhikers Guide to the Galaxy\n5. The Pirata Codex\n\n  #WritingCommunity https://t.co/xJeABPW0G3',
     'Add a caption. Any caption. #CaptionThis #WritingCommunity https://t.co/9ueKNB5TK6',
     'This. ð\x9f\x91\x87 https://t.co/EL4wqyDxVu',
     'Its nice to know that, somewhere, Galileos middle finger is as venerated as the crucifix. https://t.co/gi8uPOZF9J',
     'Name this film. Wrong answers only. https://t.co/nz0LpiPEFB',
     '[Narrated by David Attenborough]  "You can see the Karen in her native habitat: a franchise business with familiar flair. The Karen demands to see the manager, asserting dominance. Unsuccessful, the Karen resorts to violence. She is out-powered."\nhttps://t.co/ONU6ckE0sc',
     'Trump is nothing new, and neither are his supporters. https://t.co/8ZC00IGrxM',
     'Add a caption. Any caption. #CaptionThis #WritingCommunity https://t.co/6qdwvSPLFp',
     'Add a caption. Any caption.  #CaptionThis #WritingCommunity https://t.co/WrRMyOZtrp',
     'Can anyone help this fellow writers tech question? ð\x9f\x91\x87  #WritingCommunity https://t.co/M1G3gYnAxN',
     'Name this film. Wrong answers only.  #JeffGoldblum https://t.co/8ng1N0qFOX',
     'If theres one thing Mitch McConnells an expert at, its classlessness. https://t.co/UJNAKfz9xC',
     'Name this film. Wrong answers only. https://t.co/emsGLZ4Z4h',
     'What proud words would you put on Frank Costanzas headstone?  #WritingCommunity #RIPJerryStiller https://t.co/JUIRBhx8N0',
     'Current mood: https://t.co/jivFCb03vy https://t.co/S5LU7AIF2K',
     '2016: #MAGA\n2020: Transition into Greatness\n2024: Stay Happy! Its Transitioning\n2028: Reclaim the Hawaiian Islands https://t.co/4gEXG846GS',
     'Add a caption. Any caption. #CaptionThis #WritingCommunity https://t.co/0D8CYWaLqv',
     'This needs to be a movie. https://t.co/XwyvPh7QqF',
     'My God... Ray Liotta in Goodfellas DOES look like a T-Rex! https://t.co/6a8IjCso18',
     'I was waiting for someone to write this!',
     'If only!',
     'I feel like this is kinda right.',
     'Name this film. Wrong answers only. https://t.co/Pe4fPCYg4h',
     'Which film was a better addition to their franchise: Jurassic World or The Last Jedi? Explain below.  #WritingCommunity https://t.co/SyAstSWZsD',
     'IMO, "that scene" with R2 couldve been one of the best in the whole series. https://t.co/TFdJCF4Zzg',
     'Thats cute. still thinks Obama can be impeached. https://t.co/8opHy52fg1',
     'Add a caption. Any caption.  #CaptionThis #WritingCommunity https://t.co/DB9Fd5u2xW',
     'You just gave a fantastic reason for why no Christian should vote for Trump.',
     'Name this film. Wrong answers only. https://t.co/hCSJ8tPlrg',
     'Below: Trumps blueprint for rejecting the 2020 election. https://t.co/PtRwjtrfhc',
     'Add a caption. Any caption. #CaptionThis #WritingCommunity https://t.co/KkCjWh6CV5',
     '#Morocco2026 ð\x9f\x87²ð\x9f\x87¦\n\nAt the 68th Congress, Member Associations have voted in favor of the North American United bid, which will now organise the 2026 \n\n#Morocco2026 congratulates for their victory and we wish them well\n\n#TogetherForOneGoal https://t.co/3pKizFhy6O',
     '#RoadTo2026 ð\x9f\x87²ð\x9f\x87¦\n\nLast stage of our tour of #Morocco with Nas! Our special envoy went to Ouarzazate to visit the famous Noor #Morocco2026 https://t.co/i1DwptfuCP',
     '#RoadTo2026 ð\x9f\x87²ð\x9f\x87¦\nï¿¼\nItâ\x80\x99s the penultimate stage for our special correspondent Nas, who this time went to #Agadir to show us the fervor and enthusiasm of Gadiris for the 2026 World Cup #Morocco2026 https://t.co/mXk0HlYZFF',
     '#RoadTo2026 ð\x9f\x87²ð\x9f\x87¦ \n\nThe 12 host cities selected by the organising committee are within a radius of 550 km to #Casablanca - #Morocco2026 offers all football players an innovative and compact ð\x9f\x99\x8cð\x9f\x8f¼ https://t.co/eMJEjB1DmJ',
     '#Morocco2026 in 10 reasons &gt; Reason ð\x9f\x94\x9f\n\n60% of the teams at the 2026 will come from countries with a time difference of -3/+3 to #Morocco - a strategic positioning that allow fans across the world to follow their teams at ideal times \n\n#TogetherForOneGoal ð\x9f\x99\x8c https://t.co/Kjp6jMLGaK',
     '#Maroc2026 in 10 reasons &gt; Reason 9â\x83£\n\n#Morocco is the #1 financial centre in #Africa -  with a rapidly growing economy that forecasts GDP growth estimated at 4% per year until 2022 \n\n#TogetherForOneGoal ð\x9f\x99\x8c https://t.co/Hj5kg2g7uD',
     'Merci for your support of our vision for passionate, authentic ð\x9f\x99\x8c\n\n#TogetherForOneGoal ð\x9f\x87²ð\x9f\x87¦ https://t.co/DSU7nebR3f',
     'Thanks to for their support of the our vision for the 2026 \n\n#Morocco2026 ð\x9f\x87²ð\x9f\x87¦ https://t.co/pXN7h5m90S',
     '#Passion â\x9a½ \n\nDiscover the unforgettable moments lived by the children of Dar Atifl, a #Marrakech orphanage, as superstar #Morocco2026 ambassadors ð\x9f\x87§ð\x9f\x87·, LMatthaeus10 ð\x9f\x87©ð\x9f\x87ª, and ð\x9f\x87³ð\x9f\x87±\n\n#Morocco2026 ð\x9f\x87²ð\x9f\x87¦ #TogetherForOneGoal https://t.co/vpf9Mbtdis',
     '#Morocco2026 ð\x9f\x87²ð\x9f\x87¦ \n\nDiscover our vision for a compact, innovative and passionate ð\x9f\x8f\x86\n\n#TogetherForOneGoal ð\x9f\x99\x8c https://t.co/wtTQRCieRF',
     '#Russia2018 ð\x9f\x87·ð\x9f\x87º \n\nOur #AtlasLions June calendar ð\x9f\x8f\x86 \n\n15th: Morocco ð\x9f\x87²ð\x9f\x87¦-Iran ð\x9f\x87®ð\x9f\x87· (15h GMT)\n20th: Portugal ð\x9f\x87µð\x9f\x87¹-Morocco ð\x9f\x87²ð\x9f\x87¦ (12h GMT)\n25th: Spain ð\x9f\x87ªð\x9f\x87¸-Morocco ð\x9f\x87²ð\x9f\x87¦(18h GMT)\n\n#Morocco2026 ð\x9f\x99\x8cð\x9f\x8f¼ https://t.co/V40UH0KPwf',
     '#Support ð\x9f\x99\x8c\n\nâ\x80\x9cWelcome, sincerity and sharing -  #Morocco truly represents what the should be"\n\nThanks to Christophe Dugarry, World Cup Winner in 1998 with for his support of #Morocco2026 ð\x9f\x87²ð\x9f\x87¦ https://t.co/3XSHcTGqej',
     '#RoadTo2026 ð\x9f\x87²ð\x9f\x87¦ \n\nThe tenth step on our special envoy Nasâ\x80\x99 trip around Morocco sees him in #Nador, where he  has found the love of Nadoris for the 2026 ð\x9f\x99\x8cð\x9f\x8f¼\n\n#Morocco2026 ð\x9f\x87²ð\x9f\x87¦ https://t.co/kNflJPV6Fj',
     'Thanks to for his excellent #Morocco2026 presentation to Congress this morning ð\x9f\x99\x8cð\x9f\x8f¼\n\n#TogetherForOneGoal ð\x9f\x87²ð\x9f\x87¦ https://t.co/oDqs95uRXT',
     '#Morocco2026, as seen by Day 26\n\nTwo young children play #football in the countryside of Meknes â\x9a½ï¸\x8f https://t.co/Yd3fVbmDp5',
     '#Support ð\x9f\x99\x8c\n\ncaptain of #AtlasLions, has a message for you...\n\n#TogetherForOneGoal https://t.co/9OHD1Y2crN',
     '#Support ð\x9f\x99\x8c\n\n"#Morocco has everything to organise a very beautiful - I hope the country will have this opportunity." \n\na #WorldCup winner in 1998 with and a legend of discusses his passion for #Morocco2026 https://t.co/2AxExOiHli',
     'ð\x9f\x87²ð\x9f\x87¦ Your #Russia2018 Atlas Lions ð\x9f\x87²ð\x9f\x87¦\n\n#TogetherForOneGoal #Morocco2026 https://t.co/r1WELVJKaV',
     '#Support ð\x9f\x99\x8c\n\n"To organise the it would benefit all of #Morocco, and in terms of development, the whole African continent"\n\nMustapha Hadji, African Balon dOr winner in 1998 and assistant to of evokes his passion for our bid \n\n#Morocco2026 https://t.co/KOO7jYshnZ',
     '#Support ð\x9f\x99\x8c\n\n"The #Morocco2026 bid is something truly beautiful - a country so passionate about football" \n\nDiscover why #AtlasLions head coach is backing our authentic, passionate bid for the as arrive in #Russia â¬\x87ï¸\x8f https://t.co/WVGRqn1IC4',
     '#Soutien ð\x9f\x99\x8c\n\nâ\x80\x9cIn #Morocco, there is an excitement for football that isnâ\x80\x99t found anywhere else." \n\nThanks to #AtlasLions captain and star for his support of #Morocco2026 ð\x9f\x87²ð\x9f\x87¦ https://t.co/7j4X2WnfB3',
     '#Maroc2026 in 10 reasons &gt; Reason 8â\x83£\n\n#Morocco2026s infrastructure will benefit the youth of the country, both now and in the long term, with the LMS stadia providing special environments for sport to flourish across the Kingdom \n\n#TogetherForOneGoal ð\x9f\x99\x8c https://t.co/oEKcXvRcGs',
     '#Imagine2026 ð\x9f\x87²ð\x9f\x87¦ \n\nFind out how football changes the lives of women in #Morocco, through these three inspirational stories #Morocco2026 ð\x9f\x99\x8cð\x9f\x8f¼ https://t.co/3Eh4KkIz00',
     '#Morocco2026, as seen by Day 24 \n\nAmong young Moroccans #football is King â\x9a½ï¸\x8f https://t.co/9r7EQ1BleJ',
     '#Morocco2026 in 10 reasons &gt; Reason 7ï¸\x8fâ\x83£\n\nSet to open in 2018, the first high-speed line of Africa will be in #Morocco | It will connect #Tangiers to #Casablanca in only 2 hours, and will be extended to connect other major cities of the country\n\n#TogetherForOneGoal ð\x9f\x99\x8cð\x9f\x8f¼ https://t.co/6jGXs12Jut',
     '#Matchday â\x9a½ï¸\x8f\n\nThe last test before our #AtlasLions arrive at #Russia2018! men head to #Tallinn to take on #Estonia\n\n ð\x9f\x86\x9a Estonia - â\x8c\x9aï¸\x8f 16h00 GMT\nð\x9f\x8f\x9f A. Le Coq Arena (Tallinn) https://t.co/3M3nCjqzxX',
     '#Morocco2026, as seen by Day 23\n\n#Moroccan football fields are everywhere, even on the beach ð\x9f\x8f\x96 https://t.co/6DT0v20lFf',
     '#Morocco2026 in 10 reasons &gt; Reason 6â\x83£\n\nIt is estimated that #Morocco2026 will create 110,000 jobs, while the period between 2019 and 2026 will positively impact the #Moroccan economy by around US $2.7 billion ð\x9f\x99\x8cð\x9f\x8f¼ \n\n#TogetherForOneGoal ð\x9f\x87²ð\x9f\x87¦ https://t.co/RILoUtPWu5',
     '#Imagine2026 ð\x9f\x87²ð\x9f\x87¦\n\n26 Reasons Why #Morocco2026 - created by the talented applicants of \n#TogetherForOneGoal ð\x9f\x99\x8c https://t.co/CuxsqgVW3m',
     '#RoadTo2026 ð\x9f\x87²ð\x9f\x87¦ \n\n#TÃ©touan, a city alive with European accents, is the eighth stage of #Morocco2026 special envoy Nas journey across the Kingdom - watch as the local population show how ready they are to welcome supporters from around the world for the 2026 ð\x9f\x99\x8c https://t.co/nknMsnzPS2',
     '#Morocco2026, as seen by Day 22\n\nThe next generation of Casablancan footballers train in the city â\x9a½ï¸\x8f https://t.co/fu9mhdQ0zM',
     '#Passion â\x9a½\n\nThe Jemaa El Fnaa is the beating heart of #Marrakesh, where cultures come together | Watch as #Morocco2026 ambassadors and discover the very best of #football on our vibrant streets ð\x9f\x99\x8c\n\n#Morocco2026 ð\x9f\x87²ð\x9f\x87¦ https://t.co/x3J4Z0fJtX',
     '#Morocco2026 in 10 reasons &gt; Reason 5ï¸\x8fâ\x83£\n\nBy 2020, #Morocco will produce 42% of its energy from non-fossil sources, aided by the worlds largest solar station in Noor, showing once more that #Morocco is a leader in sustainability and ecology ð\x9f\x99\x8cð\x9f\x8f¼\n\n#TogetherForOneGoal ð\x9f\x87²ð\x9f\x87¦ https://t.co/4VYY7TJXeC',
     'â\x9a½ \n\nThe Jemaa El Fna is the beating heart of #Marrakesh, where cultures come together | Watch as #Morocco2026 ambassadors and discover the very best of #football on our vibrant streets ð\x9f\x99\x8c \n\n#Morocco2026 ð\x9f\x87²ð\x9f\x87¦ htt',
     '#Soutien ð\x9f\x99\x8c \n\nThanks to the #Seychelles Football Federation, through their Secretary General Georges Bibl, for their support for the Kingdom of #Morocco to host the 2026 ð\x9f\x8f\x86\n\n#Morocco2026 ð\x9f\x87²ð\x9f\x87¦ #TogetherForOneGoal https://t.co/kMEyRGi4Z0',
     '#Imagine2026 ð\x9f\x87²ð\x9f\x87¦ \n\nDiscover the history of #Moroccan football culture in this video by the talented Mohammed Elbellaoui, as part of the our nations citizen initiative \n\n#Morocco2026 ð\x9f\x99\x8c https://t.co/36sHKsb51i',
     '#RoadTo2026 ð\x9f\x87²ð\x9f\x87¦ \n\nFor his eighth episode, #Morocco2026 special Envoy Nas travelled to #Oujda, where the locals dream of hosting the best players in the world in 2026\n\n#TogetherForOneGoal ð\x9f\x99\x8c https://t.co/rouaxdnJwy',
     '#Morocco2026 in 10 reasons &gt; Reason 4â\x83£\n\n#Morocco is truly connected to the world, with 21 airports connected directly to 52 countries, and to 170 through one connection. The nations airport capacity is also set to rise to 61 million by 2025 ð\x9f\x99\x8c\n\n#TogetherForOneGoal ð\x9f\x87²ð\x9f\x87¦ https://t.co/mvFvTVg1xt',
     '#Morocco2026, as seen by Day 21\n\nThe next generation of #AtlasLions train in a Casablanca facility â\x9a½ï¸\x8f https://t.co/Vbqbq3gT6l',
     '#HostCities ð\x9f\x93\x8d #MeknÃ¨s\n\nThe Imperial City of #Meknes will have a Legacy Modular Stadium [LMS] in the stunning SaÃ¯s plain, at the foot of the middle Atlas mountains and just 15 minutes from downtown \n\n#Morocco2026 ð\x9f\x87²ð\x9f\x87¦ https://t.co/aZxIeDmy10',
     '#Media ð\x9f\x93º \n\n#Marrakesh and #Casablanca, the two cities hosting the decisive semi and final fixtures, have iconic sites to host TV Studios | In 2026, Moroccan beauty will provide the perfect backdrop for elite world football ð\x9f\x99\x8c\n\n#Morocco2026 ð\x9f\x87²ð\x9f\x87¦ https://t.co/lKMEtKbjTB',
     '#Morocco2026, as seen by Day 20 \n\nEven after nightfall, #Moroccans indulge themselves in their one true passion â\x9a½ï¸\x8f https://t.co/sqFHhFEeqV',
     '#Morocco2026 in 10 reasons &gt; Reason 3â\x83£\n\nWith its #Mediterranean climate and compact hosting plan, #Morocco2026 offers the ideal conditions for the elite of world #football to flourish on the sports greatest stage ð\x9f\x99\x8c \n\n#TogetherForOneGoal ð\x9f\x87²ð\x9f\x87¦ https://t.co/wczxeZO8Hz',
     '#RoadTo2026 ð\x9f\x87²ð\x9f\x87¦\n \nToday, our #Morocco2026 ambassadors, including and met with young people at the Stade Harti in #Marrakesh, one of the proposed training sites for the 2026  \n#TogetherForOneGoal ð\x9f\x99\x8c https://t.co/Q5Y23HdqCP',
     '#FanFests ð\x9f\x8e\x89\n\nThe #Casablanca #Corniche, with its beautiful panoramic views of the #AtlanticOcean, will welcome #WorldCup supporters for a #FanFest unlike any other - in a city renowned for its passionate fans and electric atmosphere â\x9a½ï¸\x8f \n\n#Morocco2026 ð\x9f\x87²ð\x9f\x87¦ https://t.co/DXQ7ut7dED',
     '#Support ð\x9f\x99\x8c\n \nThanks to Soufiane Touzani who joined us in #Marrakesh alongside winners Lothar Matthaeus ð\x9f\x87©ð\x9f\x87ª and Roberto Carlos ð\x9f\x87§ð\x9f\x87· to celebrate our vision for a truly passionate 2026 World Cup ð\x9f\x8f\x86\n \n#Morocco2026 ð\x9f\x87²ð\x9f\x87¦ https://t.co/EvJdcOoALI',
     '#Morocco2026, as seen by Day 18\n\nUnder the sun, young #Moroccans play #football whenever they get free time https://t.co/LPqoEMHiNo',
     '#Morocco2026 in 10 reasons &gt; Reason 2ï¸\x8fâ\x83£\n\nAll host cities are located less than an hours drive from an airport, while the most remote host cities are separated by just 75 minutes, creating ideal travel logistics for both players and fans alike ð\x9f\x99\x8cð\x9f\x8f¼\n\n#TogetherForOneGoal ð\x9f\x87²ð\x9f\x87¦ https://t.co/UuBrICFBEY',
     '#RoadTo2026 ð\x9f\x87²ð\x9f\x87¦\n\nThe seventh stop for our special envoy #Nas is #Meknes, the "Ismailia" capital of #Morocco | There he hears from visitors and locals alike about their experience of #Moroccan footballing passion â¬\x87ï¸\x8fâ¬\x87ï¸\x8f https://t.co/mbVbEDEFqI',
     '#Morocco2026, as seen by Day 17 \n\nEven on the beach, tourists and locals indulge in their true passion, #football https://t.co/rOeZgQTBA3',
     '#Support ð\x9f\x99\x8c\n\n"Morocco is a country with exceptional passion for football, and amazing infrastructure. This is an entire population that has been waiting for this honour for a very long time."\n\n#Thanks to #AtlasLions head coach for his support of #Morocco2026 ð\x9f\x87²ð\x9f\x87¦ https://t.co/mXMUmjT5ne',
     '#Morocco2026 in 10 reasons &gt; Reason 1ï¸\x8fâ\x83£\n\n #Morocco has a truly magical cultural and archaeological heritage - the Kingdom welcomed more than 11 million tourists in 2017 ð\x9f\x93¸ https://t.co/YIszYRnp7g',
     '#DidYouKnow ð\x9f\x92¡\n\nThe National Zoological Gardens of #Rabat home the largest number of #AtlasLions, a species now extinct in the wild ð\x9f¦\x81\n\n#Morocco2026 ð\x9f\x87²ð\x9f\x87¦ https://t.co/Yv9EeKlPb4',
     '#RoadTo2026 ð\x9f\x87²ð\x9f\x87¦ \n\nThanks to the President of the Pakistani Federation #Football, Faisal Saleh Hayat, for meeting to discuss our vision for #Morocco2026, a passion, accessible in a country where football is a national identity\n\n#Morocco2026 ð\x9f\x99\x8cð\x9f\x8f¼ https://t.co/3V0Q8yfOaC',
     '#Support ð\x9f\x99\x8c\n\nâ\x80\x9cThey live football, they breathe football, and this is why Morocco should host the \nThanks to legend for his support of our authentic, passionate vision for #Morocco2026\n\n#TogetherForOneGoal ð\x9f\x87²ð\x9f\x87¦ https://t.co/O2XNZSZ438',
     '#Morocco2026, as seen by Day 16\n\nYoung footballers warm-up before a match in #Casablanca https://t.co/dI4O8FkUFI',
     '#RoadTo2026 ð\x9f\x87²ð\x9f\x87¦ \n\nTask Force : â\x9c\x85 \nOn the ballot on June 13 ð\x9f\x92ª\n\n#TogetherForOneGoal https://t.co/lkhq8SmQk7',
     '#RoadTo2026 ð\x9f\x87²ð\x9f\x87¦\n\nThanks to for announcing their support for  #Morocco2026 ð\x9f\x99\x8f\n\n#TogetherForOneGoal \n\n--&gt; https://t.co/1CwAZo2AwW https://t.co/HzO4OzBFxk',
     '#Rabat ð\x9f\x93\x8d\n\nRabats Prince Moulay Abdellah Stadium has already hosted some of world sports most famous faces, welcoming the African Cup of Nations (1988), the Club World Cup (2014), and the Diamond League (2016, 2017) in its recent history ð\x9f\x99\x8c\n\n#Morocco2026 ð\x9f\x87²ð\x9f\x87¦ https://t.co/aF7EBi8XMw',
     '#Russia2018 ð\x9f\x87·ð\x9f\x87º \n\nOur #AtlasLions finish their first game of preparation with a 0-0 draw against #Ukraine | Theyâ\x80\x99re back on Monday, June 4 for a tough test against #Slovakia \n\n#Morocco2026 ð\x9f\x87²ð\x9f\x87¦ https://t.co/qEpObCg8i5',
     '#Morocco2026, as seen by Day 15\n\nThe many emotions of #football us the true way of life for Moroccans https://t.co/id2elBcyM9',
     '#Ecology ð\x9f\x8d\x83\n\nDespite homing one of the worlds biggest #solar farms, #Morocco also relies on a dozen #hydroelectric facilities, as well as fifteen wind farms, as it continues to lead the way in renewable energy across #Africa ð\x9f\x8c\x8d\n \n#Morocco2026 ð\x9f\x87²ð\x9f\x87¦ https://t.co/7za8PpBcPM',
     '#MatchDay â\x9a½ï¸\x8f\n\n#AtlasLions take on tonight as they continue their preparations ahead of the #Russia2018 ð\x9f\x99\x8c\n\nð\x9f\x86\x9a ð\x9f\x87ºð\x9f\x87¦\nâ\x8c\x9aï¸\x8f 18.00 (GMT) \nð\x9f\x8f\x9f Stade de GenÃ¨ve (Switzerland)\n\n#TogetherForOneGoal https://t.co/Enb5MIH2HV',
     'Thanks to for their warm welcome in Zurich yesterday, where the #Morocco2026 Bid team, led by shared their vision for a #passionate, #compact and #exciting 2026 with the Task Force | #TogetherForOneGoal ð\x9f\x87²ð\x9f\x87¦\n\nâ¬\x87ï¸\x8fâ¬\x87ï¸\x8f https://t.co/brF319O605',
     '#Tourism â\x98\x80ï¸\x8f\n\nRegistered as a UNESCO World Heritage Site since 2010, the iconic medina of #Chefchaouen, in the north of the country, is a truly breathtaking experience, offering a window into #Moroccoâ\x80\x99s picturesque heritage ð\x9f\x99\x8cð\x9f\x8f¼\n\n#Morocco2026 ð\x9f\x87²ð\x9f\x87¦ https://t.co/jonleyoGiV',
     '#Morocco2026, as seen by Day 14\n\nFootball, the number one sport #Morocco, is played everywhere as the streets transform into stadia https://t.co/hahAytHBnY',
     '#RoadTo2026 ð\x9f\x87²ð\x9f\x87¦\n\nThe #Casablanca #Marina will be one of the flagship projects of the citys ambitious urban development vision | Located 30 minutes from the #airport, it will include 9â\x83£ towers, a conference centre, a shopping centre and 3â\x83£ resorts ð\x9f\x99\x8c\n\n#TogetherForOneGoal https://t.co/FB1jLf7qR5',
     'ð\x9f\x99\x8cð\x9f\x87²ð\x9f\x87¦ Thankyou and welcome to #Morocco2026 ð\x9f\x99\x8cð\x9f\x87²ð\x9f\x87¦ https://t.co/YBGMqRhz4C',
     '#Stadia ð\x9f\x8f\x9fï¸\x8f\n\nThe Prince Moulay Abdellah, in Rabat, is one of #Moroccos most recognisable stadia | Built in 1983 amid an 80 hectare park, the stadium is to be renovated, adding impressive infrastructure and architecture to an already stunning landmark ð\x9f\x99\x8c\n\n#Morocco2026 ð\x9f\x87²ð\x9f\x87¦ https://t.co/o8DaUNe0aL',
     '#RoadTo2026 ð\x9f\x87²ð\x9f\x87¦ \n\nFor his sixth destination, #Morocco2026 special envoy Nas has headed to #Tangiers, where he will find out about the famous Moroccan passion for football in a city just 14km from Spain ð\x9f\x8c\x8e \n\n#TogetherForOneGoal ð\x9f\x99\x8c https://t.co/EFrlaHVjSZ',
     '#Morocco2026, as seen by Day 13\n\nIn #Morocco, football has no age limits https://t.co/9MlNOjHhlf',
     'ð\x9f\x91\x8fð\x9f\x8f¼ð\x9f\x91\x8fð\x9f\x8f¼ð\x9f\x91\x8fð\x9f\x8f¼Congratulations \n\n#Generation2026 ð\x9f\x87²ð\x9f\x87¦ https://t.co/AXbdkZuTbp',
     '#RoadTo2026 ð\x9f\x87²ð\x9f\x87¦ \n\nThe #Morocco2026 Bid Committee, led by would like to warmly thank the Chinese Football Federation (CFA) for their impeccable reception in #Beijing and the extensive meetings on the Moroccan 2026 bid and on the future of world football ð\x9f\x99\x8cð\x9f\x8f¼ https://t.co/bco2GfolMc',
     'â\x80\x9cThere is confidence from our meetings with member associations. They know Moroccan hospitality and vibrancy."\n\nHear from #Morocco2026 Bid CEO Hicham El Amrani as he talks with about our vision for the 2026 \n\nð\x9f\x87²ð\x9f\x87¦\nhttps://t.co/P8nxDayVCX',
     '#DidYouKnow ð\x9f\x92¡\n\nThe historical heart of capital city #Rabat welcomes nearly 2.5 million world-music loving visitors each year, for the incredible \n\n#Morocco2026 ð\x9f\x87²ð\x9f\x87¦ #Culture https://t.co/GGLv0PjE95',
     '#Morocco2026 and its ambassadors would like to thank the Federations of Liechtenstein and Switzerland for this weeks meetings in #Budapest, where we shared our passionate vision for the 2026 \n\n#TogetherForOneGoal ð\x9f\x87²ð\x9f\x87¦ https://t.co/NpJbSJ0QHq',
     '#DidYouKnow ð\x9f\x92¡\n\nWith an average temperature in the late afternoon and evening of 25Â°C, and a low humidity level, #Morocco offers the very best conditions for elite football to thrive ð\x9f\x99\x8c  \n\n#Morocco2026 ð\x9f\x87²ð\x9f\x87¦ https://t.co/ZvPUP9N5cF',
     '#DidYouKnow ð\x9f\x92¡\n\n#Marrakesh is the tourist capital of #Morocco, with the nations largest accommodation capacity and with a unique blend of culture and tradition | The city, with its special greeting and stunning scenery, is ready to welcome the world in 2026 \n\n#Morocco2026 ð\x9f\x87²ð\x9f\x87¦ https://t.co/J0Bct5iqf6',
     '#DidYouKnow ð\x9f\x92¡\n\n#Morocco attracts more and more foreign visitors every year - the country has experienced a 17% increase in aviation traffic between 2011 and 2017 ð\x9f\x99\x8cð\x9f\x8f¼\n\n#Morocco2026 ð\x9f\x87²ð\x9f\x87¦ https://t.co/zjZXvt5GCO',
     '#Morocco2026, as seen by Day 12\n\nThe #Moroccan people, united behind their #AtlasLions national team ð\x9f\x87²ð\x9f\x87¦ https://t.co/TY8GPMemSp',
     '#Morocco2026, as seen by Day 11\n\nA father and son share a special moment at a match in #Casablanca https://t.co/lBlvACzASW',
     '#Morocco2026, as seen by Day 10\n\nIn #Morocco, all places are conducive to the practice of #football https://t.co/j1Id2Vj1KL',
     'Support ð\x9f\x99\x8c\n\nâ\x80\x9cThe #Moroccan Bid is a Bid of the whole #African continent." \n\nRobert Kidiaba Muteba, Ambassador of #Morocco2026 and legendary keeper of TP Mazembe ð\x9f\x8f\x86\n\n#TogetherForOneGoal ð\x9f\x87²ð\x9f\x87¦ https://t.co/cfe3LYUpUq',
     '#RoadTo2026 ð\x9f\x87²ð\x9f\x87¦\n\nMany thanks to Mr. Amgalanbaatar,\nPresident of the Mongolian Football Federation, for todayâ\x80\x99s constructive meeting around #Morocco2026, a vision for a #compact and #innovative with a legacy for future generations\n\n#TogetherForOneGoal ð\x9f\x99\x8cð\x9f\x8f¼ https://t.co/nLHhx3pxRU',
     '#DidYouKnow ð\x9f\x92¡\n\nFez is considered the cultural capital of #Morocco, where a range of nationalities and traditions collide ð\x9f\x99\x8cð\x9f\x8f¼\n\n#Morocco2026 ð\x9f\x87²ð\x9f\x87¦ https://t.co/5rLCXhpfbz',
     '#RMALIV #UCLFinal ð\x9f\x8e\x89â\x9a½ï¸\x8f \n\nBravo to for victory in tonightâ\x80\x99s Final ð\x9f\x99\x8cð\x9f\x8f¼\n\nCongratulations, more importantly, to - who becomes the first #Moroccan to win the famous competition! \n\n#Morocco2026 ð\x9f\x87²ð\x9f\x87¦ https://t.co/BoDMmVZGLt',
     '#DidYouKnow ð\x9f\x92¡\n\n#Morocco2026 is a project designed for the good of all, and more than 20,000 local and international volunteers will be welcomed during the event, opening up a world of exciting opportunities ð\x9f\x99\x8cð\x9f\x8f¼\n\n#Morocco2026 ð\x9f\x87²ð\x9f\x87¦ https://t.co/NiNvXFjIM4',
     '#Morocco2026, as seen by day 9 \n\nThe next generation of footballers play in the stunning streets of Chefchaouen, in the north of the country ð\x9f\x87²ð\x9f\x87¦ https://t.co/Izrl3kqE6W',
     '#Support ð\x9f\x99\x8cð\x9f\x8f¼\n\n"I am convinced that #Morocco will offer the best conditions to the players and all lovers of #football." \n\nDavid Trezeguet Ambassador of #Morocco2026 and World Champion in 1998 with \n\n#TogetherForOneGoal ð\x9f\x87²ð\x9f\x87¦ https://t.co/5MfqH9KJTB',
     '#Morocco2026, as seen by 8 \n\nThe love for #football in #Morocco begins from the earliest of ages https://t.co/N5ZenUUStD',
     '#AfricaDay2018 \n\nToday we are celebrating World Africa Day ð\x9f\x99\x8cð\x9f\x8f¼\n\n#Morocco2026 is a Bid not only for one country, but for an entire continent, designed to leave a lasting and meaningful legacy for all of #Africa ð\x9f\x8c\x8d\n\n#DimaAfrica #Morocco2026 #AfricaDay ð\x9f\x87²ð\x9f\x87¦ https://t.co/Ve1typdDPK',
     '#DidYouKnow ð\x9f\x92¡\n\n#Morocco has one of Africaâ\x80\x99s most dynamic auto industries, doubling production in the last four years | A range of new factories, including a major facility in #Tangiers, will be opening in the next year ð\x9f\x99\x8cð\x9f\x8f¼\n\n#Morocco2026 ð\x9f\x87²ð\x9f\x87¦ https://t.co/S01Zh9AlAC',
     '#Morocco2026, as seen by 7 \n\nThree women prepare to officiate an elite football match in #Casablanca https://t.co/MTa5f8mNZm',
     '#AtlasLions captain - Ready for #Russia2018\n\n#Morocco2026 #TogetherForOneGoal ð\x9f\x87²ð\x9f\x87¦ https://t.co/HrLbEHQwqY',
     '#RoadTo2026 ð\x9f\x87²ð\x9f\x87¦ \n\nThank you for your warm welcome today and the positive discussions on our 2026 Bid, which shed light on the strengths of a compact, human and passionate built for players and fans, in a safe and tolerant country ð\x9f\x99\x8cð\x9f\x8f¼\n\n#Morocco2026 ð\x9f\x87²ð\x9f\x87¦ https://t.co/LB8Kd0IOEQ',
     '#RoadTo2026 ð\x9f\x87²ð\x9f\x87¦ \n\nThanks to for todayâ\x80\x99s constructive meeting regarding our #Morocco2026 bid - a meeting in which we shared our vision of a human, authentic, responsible and profitable ð\x9f\x99\x8cð\x9f\x8f¼\n\n#TogetherForOneGoal ð\x9f\x87²ð\x9f\x87¦ https://t.co/6SIiO83K19',
     '#RisingStar â\xad\x90ï¸\x8f\n\nA special club season ends for Hakim Ziyech, who notched 9 goals and 17 assists in and is already coveted by Europes elite clubs | The 25 year-old will be a key part of #Moroccos dreams at #Russia2018 next month ð\x9f\x99\x8c\n\n#Morocco2026 ð\x9f\x87²ð\x9f\x87¦ https://t.co/YzDBJUPRsc',
     '#RoadTo2026 ð\x9f\x87²ð\x9f\x87¦ \n\nIn his latest episode, #Morocco2026s special envoy Nas traveled to capital city #Rabat to find out just how excited both locals and tourists alike are for Morocco to host the ð\x9f\x8f\x86\n\n#TogetherForOneGoal ð\x9f\x99\x8c https://t.co/ENbbb4MTNd',
     '#Morocco2026, as seen by Day 6\n\nLocals gathered to share in their passion as take to pitch ð\x9f\x87²ð\x9f\x87¦ https://t.co/ngfdRtfWmJ',
     '#Morocco2026, as seen by Day 5\n\nTwo young women prepare to play #football in the coastal city of #Tangiers ð\x9f\x87²ð\x9f\x87¦ https://t.co/cSPgE6VSs7',
     '#DidYouKnow ð\x9f\x92¡\n\nThe Legacy Modular Stadiums can help reduce stadium capacity post- tournament by 25,000 seats, and help create stunning open plan grand stands ð\x9f\x99\x8cð\x9f\x8f¼\n\n#Morocco2026 ð\x9f\x87²ð\x9f\x87¦ https://t.co/AvrkJ3WOu8',
     '#RoadTo2026 ð\x9f\x87²ð\x9f\x87¦ \n\nWith its innovative and compact concept, #Morocco2026 will offer short travel times, meaning impeccable conditions for players, supporters and all lovers of football ð\x9f\x99\x8c\n\n#TogetherForOneGoal https://t.co/s0WbyBbKrF',
     '#RoadTo2026 \n\nThank you to the as well as its President Davor Suker, for todayâ\x80\x99s meeting | We were honored to share with you our vision for #Morocco2026, and enjoyed showing you our excellent existing and future football infrastructure \n\n#TogetherForOneGoal ð\x9f\x87²ð\x9f\x87¦ https://t.co/dhBx8jH63d',
     '#DidYouKnow ð\x9f\x92¡\n\nThe city of Oujda hosts more than 600,000 spectators each year during the International Festival of #Rai, an international cultural festival that brings the country together through music and dance\n\n#Morocco2026 ð\x9f\x87²ð\x9f\x87¦ https://t.co/y26SMCfxXW',
     '#Morocco2026 seen by Day 4\n\nSunset football on the beach of #Casablanca https://t.co/U3c6q4BUP4',
     '#Morocco2026, as seen by Day 3\n\nYoung #Moroccans, the #AtlasLions of tomorrow, play #football on the Corniche of #Tangiers https://t.co/mUrC3hUbiB',
     'Thank you for the very successul meeting held today with the membership in Johannesburg, and for your strong appreciation of #Morocco2026 - your African Bid ð\x9f\x99\x8cð\x9f\x8f¼\n\n#TogetherForOneGoal ð\x9f\x87²ð\x9f\x87¦ https://t.co/uEhbeNzLiq',
     '#Morocco2026, as seen by Day 2\n\nSupporters of show their love for the team at the Stade Mohammed V in #Casablanca â\x9a½ï¸\x8f https://t.co/CdQrI8wEua',
     '#Support ð\x9f\x99\x8c\n\n"I  believe that #Morocco, a true land of #football, will celebrate the best of the game, offering an impeccable tournament full of #authentic #passion."\n\nThanks to former winning captain Lothar MatthÃ¤us for his support of #Morocco2026 ð\x9f\x87²ð\x9f\x87¦ https://t.co/o2icW9GFPT',
     'Tomorrow at noon, #Morocco2026 will present its vision for an authentic, passionate to in Johannesburg | We cannot wait to share our message with the FAs of Southern Africa and continue to spreading #Morocco2026 worldwide ð\x9f\x99\x8c \n\n#TogetherForOneGoal ð\x9f\x87²ð\x9f\x87¦ https://t.co/UIYKLTQiBt',
     '#HappyBirthday ð\x9f\x8e\x89 \n\nÃ\x89ric Gerets, former coach of the #AtlasLions between 2010 and 2012, today celebrates his 64th birthday ð\x9f\x99\x8c\n\n#Morocco2026 ð\x9f\x87²ð\x9f\x87¦ https://t.co/VJ0hwNCdO8',
     '#Morocco2026, as seen by Day 1 \n\nChildren play #football in the streets of #Casablanca - the true #Moroccan way ð\x9f\x87²ð\x9f\x87¦\n\n#TogetherForOneGoal https://t.co/LsB9FTH31n',
     '#Morocco2026, as seen by \nEvery day to June 13th, will be sharing images from a special project curated by talented #Moroccan photographer, | Discover the everyday love and passion for #football that defines our entire nation... https://t.co/4umaxm2KV2',
     'ð\x9f\x9a¨SQUAD ANNOUNCEMENTð\x9f\x9a¨\n\nCongratulations to all of #AtlasLions as they head towards a historic summer for #Moroccan football at the #Russia2018 | #Morocco2026 ð\x9f\x87²ð\x9f\x87¦ https://t.co/hvW8lrUyKW',
     'Get well soon ð\x9f\x99\x8f\n\n#TogetherForOneGoal ð\x9f\x87²ð\x9f\x87¦#Morocco2026 https://t.co/FVtUHVJg9Z',
     'Great to see 2018 Champion and star enjoying the stunning #hostcity of #Marrakesh - after his record breaking season ð\x9f\x8f\x86\n\nWe hope to welcome you back to enjoy our special "passion for football" in 2026 ð\x9f\x99\x8c\n\n#Morocco2026 ð\x9f\x87²ð\x9f\x87¦ https://t.co/KmFVye0ykX',
     '#DidYouKnow ð\x9f\x92¡\n\nFootball is the number one sport in Morocco, with more than 73,000 licensed players nationwide ð\x9f\x99\x8c\n\nâ\x9a½ï¸\x8f #Football 73,682 \nð\x9f\x8f\x83 #Athletics 52,336 \nð\x9f¥\x8b #Karate 42,500 \nð\x9f¤¾ð\x9f\x8f¾ #Handball 29,000 \nð\x9f\x91\x8a #Taekwondo 28,000 \n\n#Morocco2026 ð\x9f\x87²ð\x9f\x87¦ https://t.co/LXFxzS3U21',
     '#Legend ð\x9f\x99\x8c\n\n"#Morocco has the ability to organise the - for me, that is a certainty"\n\nThanks to winner and former and legend Laurent Blanc for his positive words for #Morocco2026\n\n#TogetherForOneGoal ð\x9f\x87²ð\x9f\x87¦ https://t.co/rAMXZXUPRF',
     '#HostCities ð\x9f\x93\x8d#Casablanca \n\nThe coastal citys skyline will soon be home to the stunning Grand ThÃ©Ã¢tre de Casablanca, before the Casablanca Finance City is established in 2020, further establishing Casablanca as an economic and cultural metropolis ð\x9f\x99\x8c \n\n#Morocco2026 ð\x9f\x87²ð\x9f\x87¦ https://t.co/8xdCvJreiI',
     '#HostCities ð\x9f\x93\x8d #Casablanca\n \nCasablanca, as the economic centre of the Kingdom, is particularly cosmopolitan - the city gave its name to the iconic film with Humphrey Bogart and Ingrid Bergman and welcomes cultural and artistic events in abundance ð\x9f\x99\x8c \n\n#Morocco2026 ð\x9f\x87²ð\x9f\x87¦ https://t.co/zU2jdhGsIM',
     '#TogetherForOneGoal â\x9a½ \n\nThis time in #Moroccos capital of tourism, #Marrakesh, Nas is meeting with travelers and locals alike to get a flavour for the worlds appetite for #Morocco2026 ð\x9f\x87²ð\x9f\x87¦\n\nâ¬\x87ï¸\x8fâ¬\x87ï¸\x8f https://t.co/wVpBz1Orz9',
     '#Passion â\x9a½ \n\nICYMI: Saturday saw Ittihad Tangier crowned #Champions of the #BotolaPro for the first time ever | Discover here the very best of a momentous and passionate day in the history of Moroccan football ð\x9f\x94¥\n\n#Morocco 2026 ð\x9f\x87²ð\x9f\x87¦ https://t.co/cKtes8OliS',
     '#DidYouKnow ð\x9f\x92¡\n\n91% of the #Moroccan population believes that the organization of the would have a positive impact on national sport at all levels &gt;&gt; (Poll - February 2018)\n\n#Passion ð\x9f\x87²ð\x9f\x87¦ #Morocco2026 https://t.co/02e0lqpQxn',
     '#DidYouKnow ð\x9f\x92¡\n\n#Morocco2026 already has outstanding marketing potential | Today, the Kingdom is one of the most dynamic sponsorship markets in Africa, while its sweetspot location will provide a perfect platform for European broadcasters and sponsors alike ð\x9f\x99\x8c \n\n#Morocco2026 https://t.co/S68E0eKxnw',
     'Congratulation to #Moroccan on being named in the Ligue 2 Team of the Year at the #TropheesUNFP ð\x9f\x99\x8c\n\n#Morocco2026 ð\x9f\x87²ð\x9f\x87¦ https://t.co/cXPbzb0qk8',
     '#HostCities ð\x9f\x93\x8d #Oujda \n\nOujda Airport - one of the busiest in the country - has the capacity to welcome nearly 3,000,000 passengers per year, and will offer fans and players alike a warm #Moroccan welcome in 2026 ð\x9f\x99\x8cð\x9f\x8f¼\n\n#Morocco2026 ð\x9f\x87²ð\x9f\x87¦ https://t.co/E2ASCyGoZx',
     '#DidYouKnow ð\x9f\x92¡\n\n#Morocco recently hosted the Kitesurfing World Championships, an event which stimulated numerous environmental awareness operations, including a range of beach clean-up sessions ð\x9f\x8d\x83\n\n#Morocco2026 ð\x9f\x87²ð\x9f\x87¦ https://t.co/gCq977MXNp',
     '#Stadiums ð\x9f\x93\x8d #Oujda \n\nThe Oujda stadium will be one of the most eco-friendly stadiums in Africa, generating more electricity than it consumes ð\x9f\x8d\x83\n\n#Morocco2026 ð\x9f\x87²ð\x9f\x87¦ https://t.co/d5XCIz9sMJ',
     '#Congratulations ð\x9f\x8f\x86 to Ittihad #Tangier as they are crowned champions of the #Moroccan #Botola for the first time ever - incredible scenes ð\x9f\x99\x8cð\x9f\x8f¼\n\n#Morocco2026 ð\x9f\x87²ð\x9f\x87¦ #Passion https://t.co/F0INhxso9b',
     '#DidYouKnow ð\x9f\x92¡ \n\nThe 2nd weekend of May is the occasion to celebrate the Feast of Flowers, in El Kelaa Mâ\x80\x99Gouna | The event attracts thousands of visitors each year, all enjoying a programme of songs, folk dances and parades #Culture ð\x9f\x99\x8cð\x9f\x8f¼\n\n#Morocco2026 ð\x9f\x87²ð\x9f\x87¦ https://t.co/n0z9u3kbiE',
     '#TogetherForOneGoal ð\x9f¥\x87\n\nAt a Sports Day, on 11 May, the youth of Imelchil came together to support our visionary Bid for #Morocco2026 | Hundreds of children gathered and took part in the "Sport for all Caravan" - an initiative developed by \n\n#FootballForAll ð\x9f\x87²ð\x9f\x87¦ https://t.co/O8Wqu8Gdmr',
     '#DidYouKnow ð\x9f\x92¡\n\nSix of the ten most flourishing economies in the world in 2017 were #African? #Morocco2026 is a gateway to one of the worldâ\x80\x99s most dynamic continents ð\x9f\x99\x8cð\x9f\x8f¼\n\n#DimaAfrica ð\x9f\x87²ð\x9f\x87¦ #TogetherForOneGoal https://t.co/4s0FmtSg1x',
     '#DidYouKnow ð\x9f\x92¡\n\nThree #Moroccan clubs have taken part in the Club World Cup in the past decade:\n\nð\x9f\x91\x89ð\x9f\x8f¼ in 2013\nð\x9f\x91\x89ð\x9f\x8f¼ MA TÃ©touan in 2014 \nð\x9f\x91\x89ð\x9f\x8f¼ in 2017\n\n#Morocco2026 ð\x9f\x87²ð\x9f\x87¦ #Authenticity https://t.co/XmI4OInjmH',
     '#HappyBirthday to #AtlasLions legend and star goalkeeper ð\x9f\x99\x8cð\x9f\x8f¼\n\n#Morocco2026 ð\x9f\x87²ð\x9f\x87¦ https://t.co/osMdrJiA8T',
     '#HostCities | #Rabat ð\x9f\x93\x8d\n\nThe National Football Centre Maamora is undergoing impressive renovation, creating an incredible space for the benefit of football throughout both #Morocco and the entire #African continent ð\x9f\x99\x8cð\x9f\x8f¼\n\n#Morocco2026 ð\x9f\x87²ð\x9f\x87¦ https://t.co/fyuPtlYAlo',
     '#HostCities | #Rabat ð\x9f\x93\x8d\n\nAs the media continue their tour to #Rabat, they head to the National Football Centre Maamora, where President Fouzi Lekjaa discusses the upgrade plans for the state-of-the-art facility ð\x9f\x99\x8cð\x9f\x8f¼\n\n#Morocco2026 ð\x9f\x87²ð\x9f\x87¦ https://t.co/A8P6lG16Lz',
     'Congratulations to #AtlasLions ð\x9f\x87²ð\x9f\x87¦captain and star on winning the #CoppaItalia ð\x9f\x8f\x86\n\n#TogetherForOneGoal \n#Morocco2026 ð\x9f\x99\x8cð\x9f\x8f¼ https://t.co/i9qeZvfDPw',
     '#RoadTo2026 | #Tangiers ð\x9f\x93\x8d\n\nAs media continue to tour #Morocco, they head to the TGV Station in Tangiers, which from October of this year will operate Africaâ\x80\x99s first ever high-speed train ð\x9f\x9a\x85\n\n#Morocco2026 ð\x9f\x87²ð\x9f\x87¦ https://t.co/KaXsr9WwH0',
     '#HostCities | #Tangiers ð\x9f\x93\x8d\n\nAt the most northern point of the continent, African culture blends seamlessly with Arabic and European imagery to create the incredibly rich diversity of #Tangiers, another of our incredible host cities ð\x9f\x99\x8cð\x9f\x8f¼\n\n#Morocco2026 ð\x9f\x87²ð\x9f\x87¦ https://t.co/3KvxFrk6Ah',
     '#Stadiums | #Tangiers ð\x9f\x93\x8d\n\nThe stunning Stade Ibn Batouta in the city of artists, Tangiers, sits just 14km from Europe and will host 65,000 people for a range of games, including a quarter final, in 2026 ð\x9f\x99\x8cð\x9f\x8f¼\n\n#Morocco2026 ð\x9f\x87²ð\x9f\x87¦ https://t.co/zw5BE7obs8',
     '#HostCities | #Tangiers ð\x9f\x93\x8d\n\n#Morocco is a country rich in football history, and as media arrive at the Stade Ibn Batouta they are welcomed by a timeline of #Morocco success, both past and present ð\x9f\x99\x8cð\x9f\x8f¼\n\n#Morocco2026 ð\x9f\x87²ð\x9f\x87¦ https://t.co/g9punxG3qh',
     '#Marrakesh â\x9e¡ï¸\x8f #Tangiers\n\nAfter an early start in #Marrakesh, the media now head to #Tangiers to visit a city just 14km from Europe, and tour itâ\x80\x99s stunning host stadium, Stade Ibn Batouta ð\x9f\x99\x8cð\x9f\x8f¼\n\n#Morocco2026 ð\x9f\x87²ð\x9f\x87¦ https://t.co/Mz0stcKJNj',
     '#Marrakesh ð\x9f\x93\x8d\n\nBridging the old city and the new, the   Bahia Palace sits at the heart of #Marrakesh medina and gives a window into the stunning culture and beauty of the nation ð\x9f\x99\x8cð\x9f\x8f¼\n\n#Morocco2026 ð\x9f\x87²ð\x9f\x87¦ https://t.co/YCIOtfgyK9',
     '#DidYouKnow ð\x9f\x92¡\n\nAt least 60% of the nations that will compete at the 2026 are located within a -3/+3 timezone of Morocco, providing a lucrative sweetspot location for #European and #African broadcasters  \n\n#Morocco2026 ð\x9f\x87²ð\x9f\x87¦ https://t.co/KELxw3f5j3',
     'Great to have the unanimous support of the Organisation of Islamic Cooperation a body formed from 57 Member Nations from across the globe ð\x9f\x99\x8cð\x9f\x8f¼\n\n#Morocco2026 ð\x9f\x87²ð\x9f\x87¦ https://t.co/UbgQs2YpvZ',
     '#GrandStadeDeMarrakesh ð\x9f\x93\x8d\n\nNext stop on the tour is Grand Stade de Marrakesh, which will host a range of matches in 2026, including one of the semi-finals | #Morocco is ready to welcome the elite of world football ð\x9f\x99\x8cð\x9f\x8f¼\n\n#Morocco2026 ð\x9f\x87²ð\x9f\x87¦ https://t.co/KEziLqws95',
     '#StadeElHarti | #Marrakesh ð\x9f\x93\x8d\n\nThe #StadeElHarti will serve the worldâ\x80\x99s elite teams in 2026, and has already welcomed the likes of for the #FIFAClubWorldCup ð\x9f¥\x87\n\n#Morocco2026 ð\x9f\x87²ð\x9f\x87¦ https://t.co/NFfBYqjKt8',
     '#StadeElHarti | #Marrakesh ð\x9f\x93\x8d\n\nAs the ð\x9f\x8c\x8dâ\x80\x99s media tour #Morocco this week, we stop at the #StadeElHarti, one of #Marrakeshâ\x80\x99s key proposed training venues for 2026 ð\x9f\x87²ð\x9f\x87¦\n\n#Morocco2026 ð\x9f\x99\x8cð\x9f\x8f¼ https://t.co/sSIwiEerTg',
     '#LÃ©gende ð\x9f\x8f\x86\n\na pillar of the defense of the #AtlasLions with 59 caps and 3 goals, was part of the finalist team of the 2004 AFCON \n\n#Morocco2026 ð\x9f\x87²ð\x9f\x87¦ https://t.co/3xyCHkmTtR',
     '#TogetherForOneGoal ð\x9f\x87²ð\x9f\x87¦ \n\nNas has given himself a mission: to show the enthusiasm and excitement for football of Moroccaneach across all of our host cities! Check out the first episode in #Casablanca, for the derby ð\x9f\x99\x8cð\x9f\x8f¼\n\n#Morocco2026 ð\x9f\x87²ð\x9f\x87¦ https://t.co/JnuGOBRGZw',
     '#Economy\n\nThe unique character of our proposed cities, such as #Marrakech and #Casablanca, will serve as a special opportunity for business partners, media and broadcasters alike in 2026\n\n#Morocco2026 ð\x9f\x87²ð\x9f\x87¦ https://t.co/mvPZAYTZkd',
     'Congratulations to ð\x9f\x87²ð\x9f\x87¦ \n\n#Generation2026 https://t.co/So4eBjR2Uj',
     '#HostCities ð\x9f\x93\x8d #ElJadida \n\nRich in tradition, El Jadida is now oriented towards the future, with its stunning Mazagan seaside area ð\x9f\x99\x8cð\x9f\x8f¼\n\n#Morocco2026 ð\x9f\x87²ð\x9f\x87¦ https://t.co/LWIXvH2w4Q',
     '#Support ð\x9f\x99\x8c\n\nâ\x80\x9cMany countries around the world will not need to get up at impossible times to watch matches" \n\ndiscusses his reasons for supporting #Morocco2026 ð\x9f\x87²ð\x9f\x87¦\n\n&gt;&gt;&gt; https://t.co/MKiQG27IsP',
     '#DidYouKnow ð\x9f\x92¡ \n\nToday is the first day of the Group Stage of the #AfricanChampionsLeague | The holders of the prestigious title are the Moroccan club who claimed victory in 2017 against Egyptian side ð\x9f\x99\x8cð\x9f\x8f¼\n\n#Morocco2026 ð\x9f\x87²ð\x9f\x87¦ https://t.co/ry3gAHwUy7',
     '#LÃ©gende ð\x9f\x8f\x86 \n\npaid homage to the career of iconic #Moroccan footballer Abdellah El Antaki, at a ceremony yesterday | He took part in the famous #Morocco v #Spain match of 1961 ð\x9f\x99\x8cð\x9f\x8f¼ https://t.co/hGgbGMSJcc',
     '#TogetherForOneGoal\n\nPresident Fouzi Lekjaa and the #Morocco2026 Bid Committee are grateful to for his warm welcome and for appreciating the concept and strengths of #Morocco2026 ð\x9f\x87²ð\x9f\x87¦ https://t.co/DRdtRMPojc',
     '#TogetherForOneGoal  \n\nToday, #Morocco2026 had the honour of sharing our vision with the Nordic Football Federations in Copenhagen \n\nAn authentic, profitable #Morocco2026 offers the best of #football just 14 km from our #European neighbours\n\n#Morocco2026 ð\x9f\x87²ð\x9f\x87¦ https://t.co/Nqh2inFOsA',
     '#TogetherForOneGoal ð\x9f\x99\x8c\n\n#DidYouKnow ð\x9f\x92¡ #Morocco2026 has today launched its stunning #Instagram account, which will go on to illustrate 26 iconic moments of #Moroccan football ð\x9f\x93¸\n\nDiscover for yourself the first four frescoes NOW &gt;&gt;&gt; https://t.co/0DtOJBNfUt https://t.co/XEnPSO8VW3',
     '#DimaAfrica ð\x9f\x8c\x8d\n\n#Moroccos bid for the 2026 is a truly African project, symbolizing a new era of commitment by #Morocco to #Africa, following the return of the country to the African Union in 2017 ð\x9f\x99\x8c\n\n#Morocco2026 ð\x9f\x87²ð\x9f\x87¦ https://t.co/6KKUkxzwKg',
     '#HostCities ð\x9f\x93\x8d #ElJadida\n\nClose to the city of #Casablanca, El Jadida welcomes almost 500,000 visitors each year to the #Jawhara Festival, which sees nearly 500 #Moroccan artists take to the stage for a celebration of #music and #culture ð\x9f\x99\x8c\n\n#Morocco2026 ð\x9f\x87²ð\x9f\x87¦ https://t.co/aDbILj3MRQ',
     '#RoadTo2026 ð\x9f\x87²ð\x9f\x87¦ #Stadiums ð\x9f\x8f\x9f\n\nOur 6 Legacy Modular Stadiums go even further than the traditional modular concept:\n\nð\x9f\x8d\x83 100% environmentally friendly\nð\x9f\x92° Shared designs mean lower costs\nð\x9f\x91¶ Sustainable legacies that will benefit future generations, in both #Morocco and #Africa https://t.co/daVeq3YcNJ',
     '#Supportð\x9f\x99\x8cð\x9f\x8f¼\n\nThanks to the President of the #Myanmar Football Federation ð\x9f\x87²ð\x9f\x87², Zaw Zaw, for his support of our #Morocco2026 Bid ð\x9f\x87²ð\x9f\x87¦ \n\n&gt;&gt; https://t.co/gZnPlGTV9T https://t.co/rn5RSydsL1',
     '#RisingStar ð\x9f\x8f\x85 \n\nCongratulations to #RealMadrid star who last night qualified his third consecutive final ð\x9f\x99\x8c\n\n#Morocco2026 ð\x9f\x87²ð\x9f\x87¦ https://t.co/u3OcaSRxq2',
     '#HostCities ð\x9f\x93\x8d #ElJadida\n\n#DidYouKnow ð\x9f\x92¡ The #Morocco2026 Host City of #ElJadida was formerly known as Mazayan, and is a registered World Heritage Site | #Morocco2026 will be a with culture at every turn ð\x9f\x99\x8c\n\n#TogetherForOneGoal ð\x9f\x87²ð\x9f\x87¦ https://t.co/jXvtsFsTnT',
     '#DidYouKnow ð\x9f\x92¡\n\nIn 2017, #Morocco welcomed a record number of tourists, with more than 11 million visitors from around the globe descending upon the Kingdom | Known for its hospitality, #Morocco ð\x9f\x87²ð\x9f\x87¦ is ready to welcome the world in 2026\n\n#TogetherForOneGoal ð\x9f\x99\x8c https://t.co/vMaeOTHZf1',
     '#DidYouKnowð\x9f\x92¡\n\nThe street-art festival "Jidar, toiles de rue was held in #Rabat last month, between April 16th to 22th | A dozen new frescoes by internationally acclaimed artists now adorn the nations capital #Culture ð\x9f\x99\x8c\n\n#Morocco2026 ð\x9f\x87²ð\x9f\x87¦ https://t.co/fIfBCYPQ8m',
     '#AtlasLions boss ð\x9f\x97£ï¸\x8f\n\n"[Winning] would be something exceptional - I am very confident that #Morocco can be a superb host of the 2026 \nWatch the full interview with here: https://t.co/JvBAA9HDmh  \n\n#TogetherForOneGoal ð\x9f\x87²ð\x9f\x87¦ https://t.co/4y8Hxo3rgF',
     '#Support ð\x9f\x99\x8cð\x9f\x8f¼\n\n#Morocco is a Bid for all of #Africa, and our #Senegalese star ambassador El Hadji Diouf is calling on all footballer lovers from around the world to unite behind #Morocco2026 ð\x9f\x87²ð\x9f\x87¦\n\n#TogetherForOneGoal https://t.co/DwLwqEJDN4',
     '#HappyBirthday ð\x9f\x8e\x89\n\n...to Abdelaziz Souleimani, who is 60 years old today | The former #AtlasLions star was the pride of #Morocco during our famous run to the Round of 16 at #Mexico86\n\n#Morocco2026 ð\x9f\x87²ð\x9f\x87¦ https://t.co/JCW5TFOPPk',
     '#ICYMI: This weekend, #Morocco2026 supported thousands of women as they took part in the Womenâ\x80\x99s Race For Victory in #Rabat | Developed by the race celebrates womenâ\x80\x99s sport throughout the nation\n\n#Morocco2026 ð\x9f\x87²ð\x9f\x87¦ #TogetherForOneGoal https://t.co/Cli6kHxS1Q',
     '#DidYouKnow ð\x9f\x92¡\n\nThousands of women of all ages participated in the Womenâ\x80\x99s Race of Victory today, which together with Nezha Bidouane was supported by #Morocco2026 \n\n#TogetherForOneGoal ð\x9f\x87²ð\x9f\x87¦ https://t.co/HtHtHqkPLF',
     '#DidYouKnow ð\x9f\x92¡\n\nThe #AtlasLions are the pride of #Morocco, with national team matches regularly attracting TV audiences of up to 70% of the entire population ð\x9f\x99\x8cð\x9f\x8f¼\n\n#Morocco2026 ð\x9f\x87²ð\x9f\x87¦ #Unity https://t.co/8qVeTPpKVc',
     '#DidYouKnow ð\x9f\x92¡\n\nThe Moroccan high-speed train line, the first of its kind in #Africa, will create more than 2,500 jobs in the country ð\x9f\x9a\x85\n\n#Morocco2026 ð\x9f\x87²ð\x9f\x87¦ https://t.co/m02You2a30',
     '#HostCities ð\x9f\x93\x8d #Nador \n\n#DidYouKnow ð\x9f\x92¡Nador is going through a major restructuring project, and is oriented towards eco-tourism and sustainable growth, combining seamlessly the city with nature\n\n#Morocco2026 ð\x9f\x87²ð\x9f\x87¦ https://t.co/f5Us3C0Vmj',
     '#Tribute ð\x9f\x99\x8f \n\nOn 27 April 1993, 25 years ago today, the Zambian Air Force Flight 319 crashed in the Atlantic Ocean with the Zambian national football team onboard.\n\nToday, we remember the passengers and crew members present that day.\n\n#DimaAfrica https://t.co/rcQ3FIlkNE',
     '#Talent ð\x9f¥\x87\n\nA surprise winner with Zambia in 2012, and champion in 2015 with CÃ´te dâ\x80\x99Ivoire, is the only manager to have won the #AFCON with two different nations | This summer he will lead the #AtlasLions ð\x9f\x87²ð\x9f\x87¦ to #Russia2018 \n\n#Morocco2026 ð\x9f\x99\x8cð\x9f\x8f¼ https://t.co/0FZjY28AEo',
     '#DidYouKnow ð\x9f\x92¡\n\n#Morocco is the #ï¸\x8fâ\x83£1ï¸\x8fâ\x83£ tourist destination in #Africa | Rich in mountain ranges, stunning beaches, magical cities and eclectic culture, #Morocco will offer something for everyone in 2026 ð\x9f\x99\x8cð\x9f\x8f¼\n\n#Morocco2026 ð\x9f\x87²ð\x9f\x87¦ https://t.co/kInozSXItM',
     '#RoadTo2026 ð\x9f\x87²ð\x9f\x87¦ #FanFests ð\x9f\x8e\x89 \n\nThe #Morocco2026 #FanFests will be open from 10:00am to midnight and in addition to the 80 elite level games, supporters will be able to enjoy local culture, leisure activities and free sports events at all locations ð\x9f\x99\x8cð\x9f\x8f¼ \n\n#Morocco2026 ð\x9f\x87²ð\x9f\x87¦ https://t.co/Grs33ErCm2',
     '#DimaAfrica ð\x9f\x8c\x8d\n\n#Morocco2026 will not only catalayse our nation, but will inject the entire #African continent with energy and enthusiasm, and will drive development and growth for all ð\x9f\x99\x8cð\x9f\x8f¼\n\n#Morocco2026 ð\x9f\x87²ð\x9f\x87¦ #TogetherForOneGoal https://t.co/BrqwHnHcYH',
     '#Welcome ð\x9f\x99\x8cÂ\xa0\n\nThe growth of #Moroccos hotel capacity has risen by 7% every year since 2003, meaning that fans and players from around the world will be treated to a truly special experience in 2026\n\n#Morocco2026 ð\x9f\x87²ð\x9f\x87¦ https://t.co/YMho6MehsF',
     '#TropheeHassanII â\x9b³ï¸\x8f \n\nCongratulations to and for their victory at the in #Rabat | The end of a magnificent weekend of elite world sport in #Morocco, a nation ready for 2026\n\n#Morocco2026 ð\x9f\x87²ð\x9f\x87¦ https://t.co/SP9QLaY3zj',
     '#Culture ð\x9f\x99\x8c\n\n#Rabats stunning which opened its doors in 2014, preserves the heritage of Moroccan art and culture | In 2018, the museum celebrated the rich influence of Mediterranean art in #Morocco with works by #Dali, #Matisse, and #Braque\n\n#Morocco2026 ð\x9f\x87²ð\x9f\x87¦ https://t.co/T2ZuNHNGxb',
     '#Tribute ð\x9f\x99\x8fð\x9f\x8f½ \n\nHenri Michel, former coach of the #AtlasLions, who led the Moroccan National Team at the 1998 died this morning at the age of 70 \n\nWe express our deepest condolences to his family and to his relatives https://t.co/YgGZkkXP82',
     ...]




```python
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

data_words = list(sent_to_words(data))

print(data_words[:1])
```

    [['truly', 'is']]
    


```python
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100) 
```

    C:\Users\Hp\Anaconda3\lib\site-packages\gensim\models\phrases.py:494: UserWarning: For a faster implementation, use the gensim.models.phrases.Phraser class
      warnings.warn("For a faster implementation, use the gensim.models.phrases.Phraser class")
    


```python
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# See trigram example
print(trigram_mod[bigram_mod[data_words[0]]])
```

    ['truly', 'is']
    


```python
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out
```


```python
# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)

# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# python3 -m spacy download en
nlp = spacy.load('en', disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

print(data_lemmatized[:1])
```

    [['truly']]
    


```python
# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)

# Create Corpus
texts = data_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# View
print(corpus[:1])
```

    [[(0, 1)]]
    


```python
id2word[0]
```




    'truly'




```python
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=4, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
```


```python
#Print the Keyword in the 10 topics
print(lda_model.print_topics())
doc_lda = lda_model[corpus]
```

    [(0, '0.025*"thank" + 0.021*"today" + 0.017*"good" + 0.017*"year" + 0.014*"see" + 0.014*"great" + 0.013*"day" + 0.013*"take" + 0.011*"http" + 0.010*"read"'), (1, '0.086*"co" + 0.054*"https" + 0.023*"covid" + 0.017*"support" + 0.014*"say" + 0.013*"go" + 0.012*"need" + 0.012*"get" + 0.011*"country" + 0.011*"woman"'), (2, '0.021*"people" + 0.019*"work" + 0.019*"new" + 0.018*"make" + 0.011*"well" + 0.010*"global" + 0.010*"must" + 0.009*"community" + 0.009*"research" + 0.008*"first"'), (3, '0.032*"amp" + 0.020*"time" + 0.014*"learn" + 0.014*"call" + 0.012*"want" + 0.011*"report" + 0.010*"life" + 0.009*"share" + 0.008*"education" + 0.008*"case"')]
    


```python
social=["thank","today","good","year","see","great","day","take","read"]
economy=["people","work","new","make","well","global","must","community","research","first"]
health=["covid","support",'say',"go","need","get","country","woman"]
cultural=["time","learn","call","want","report","life","share","education","case"]
```


```python

```


```python
fg=['i','88']
yu='i am a dog'
matchers = ['abc','def']
matching = [s for s in fg if(s in yu)]
if matching:
    print (len(matching))
```

    1
    


```python
social=["thank","today","good","year","see","great","day","take","read"] #0
economy=["people","work","new","make","well","global","must","community","research","first"] #1
health=["covid","support",'say',"go","need","get","country","woman"] #2
cultural=["time","learn","call","want","report","life","share","education","case"] #3 or education
df.columns
```




    Index(['Unnamed: 0', 'id', 'created_at', 'source', 'original_text',
           'clean_text', 'sentiment', 'polarity', 'subjectivity', 'lang',
           'favorite_count', 'retweet_count', 'original_author',
           'possibly_sensitive', 'hashtags', 'user_mentions', 'place',
           'place_coord_boundaries', 'user', 'Unnamed: 0.1'],
          dtype='object')




```python
vote=[]
for i in df.index:
    x=str(df.at[i,'original_text'])
    #print (x)
    matching=[s for s in x if(s in social)]
    if matching:
        vote.append(0)
    else:
        matching=[s for s in x if(s in economy)]
        if matching:
            vote.append(1)
        else:
            matching=[s for s in x if(s in health)]
            if matching:
                vote.append(2)
            else:
                matching=[s for s in x if(s in cultural)]
                if matching:
                    vote.append(3)
                else:
                    vote.append(random.randint(0, 3))
```


```python
len(vote)
```




    377694




```python
df['topic']=vote

```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>id</th>
      <th>created_at</th>
      <th>source</th>
      <th>original_text</th>
      <th>clean_text</th>
      <th>sentiment</th>
      <th>polarity</th>
      <th>subjectivity</th>
      <th>lang</th>
      <th>...</th>
      <th>retweet_count</th>
      <th>original_author</th>
      <th>possibly_sensitive</th>
      <th>hashtags</th>
      <th>user_mentions</th>
      <th>place</th>
      <th>place_coord_boundaries</th>
      <th>user</th>
      <th>Unnamed: 0.1</th>
      <th>topic</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1.2869e+18</td>
      <td>Sat Jul 25 05:24:59 +0000 2020</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>@DewloUKnow @Amy_Siskind Truly is</td>
      <td>Truly</td>
      <td>Sentiment(polarity=0.0, subjectivity=0.0)</td>
      <td>0.0000</td>
      <td>0</td>
      <td>en</td>
      <td>...</td>
      <td>0</td>
      <td>smora75</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>DewloUKnow, Amy_Siskind</td>
      <td>NaN</td>
      <td>Whittier</td>
      <td>smora75</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1.28689e+18</td>
      <td>Sat Jul 25 05:12:32 +0000 2020</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>@alternabirth @chrissyteigen My information co...</td>
      <td>My information comes helping physicians fill d...</td>
      <td>Sentiment(polarity=0.0, subjectivity=0.0666666...</td>
      <td>0.0000</td>
      <td>0.0666667</td>
      <td>en</td>
      <td>...</td>
      <td>0</td>
      <td>smora75</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>alternabirth, chrissyteigen</td>
      <td>NaN</td>
      <td>Whittier</td>
      <td>smora75</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>1.28686e+18</td>
      <td>Sat Jul 25 03:03:17 +0000 2020</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>@VioletRiotGames Yes my Dad taught me</td>
      <td>Yes Dad taught</td>
      <td>Sentiment(polarity=0.0, subjectivity=0.0)</td>
      <td>0.0000</td>
      <td>0</td>
      <td>en</td>
      <td>...</td>
      <td>0</td>
      <td>smora75</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>VioletRiotGames</td>
      <td>NaN</td>
      <td>Whittier</td>
      <td>smora75</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>1.2868e+18</td>
      <td>Fri Jul 24 22:51:53 +0000 2020</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>@odranwaldo I believe my brain is loud enough ...</td>
      <td>I believe brain loud enough already thanks</td>
      <td>Sentiment(polarity=0.10000000000000002, subjec...</td>
      <td>0.1000</td>
      <td>0.5</td>
      <td>en</td>
      <td>...</td>
      <td>0</td>
      <td>smora75</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>odranwaldo</td>
      <td>NaN</td>
      <td>Whittier</td>
      <td>smora75</td>
      <td>NaN</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>1.2868e+18</td>
      <td>Fri Jul 24 22:50:36 +0000 2020</td>
      <td>&lt;a href="http://twitter.com/download/android" ...</td>
      <td>@DrPnygard @MollyJongFast @SmileItsNathan Yes ...</td>
      <td>Yes precisely I zoomed original</td>
      <td>Sentiment(polarity=0.3875, subjectivity=0.775)</td>
      <td>0.3875</td>
      <td>0.775</td>
      <td>en</td>
      <td>...</td>
      <td>0</td>
      <td>smora75</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>DrPnygard, MollyJongFast, SmileItsNathan</td>
      <td>NaN</td>
      <td>Whittier</td>
      <td>smora75</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>




```python
user=df.user.unique()
topic0=[]
topic1=[]
topic2=[]
topic3=[]
for i in user:
    topic0=len(df.loc[(df['user']==i) &  (df['topic']==0)][['user','topic']])
    topic1=len(df.loc[(df['user']==i) &  (df['topic']==1)][['user','topic']])
    topic2=len(df.loc[(df['user']==i) &  (df['topic']==2)][['user','topic']])
    topic3=len(df.loc[(df['user']==i) &  (df['topic']==3)][['user','topic']])
    dict1={'user':i,'topic0':topic0,'topic1':topic1,'topic2':topic2,'topic3':topic3}
    df1=pd.DataFrame(dict1, index=[0])
    try:
        A=pd.concat([df1,A])
    except NameError:
        A=df1
```


```python
#A.to_csv('df_topic.csv')
```


```python

```


```python
x=A[['topic0','topic1','topic2','topic3']]
scaler=StandardScaler().fit(x)
newx=scaler.transform(x)
kmax = 10
sil=[]

# dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
for k in range(2, kmax+1):
    kmeans = KMeans(n_clusters = k).fit(newx)
    labels = kmeans.labels_
    sil.append(silhouette_score(newx, labels, metric = 'euclidean'))

rangex=range(2,kmax+1)
fig,ax=plt.subplots()
ax.plot(rangex,sil)
ax.set_xlabel('No of Cluster', size=10)
ax.set_ylabel('Silhouettescore', size=10)
fig.savefig('sil plot.jpg')
```


![png](Twittercommunities_files/Twittercommunities_57_0.png)



```python
kmeans = KMeans(n_clusters = 2).fit(x)
```


```python

```


```python

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
# using two cluster centers:
kmeans = KMeans(n_clusters=2)
kmeans.fit(x)
ax.scatter(A['topic0'], A['topic2'], c=kmeans.labels_, marker='o')
ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],marker='^',c='r',s=[120,150])
ax.set_xlabel('Social', size=10)
ax.set_ylabel('Economy',size=10)
fig.savefig('cluster.jpg')
```


![png](Twittercommunities_files/Twittercommunities_60_0.png)



```python
predict=kmeans.predict(x)
A['cluster']=predict
#A.to_csv('A.to_csv')
```


```python
del A['Unnamed: 0']
del df['Unnamed: 0']
```


```python
newdf=pd.merge(A,df,on='user')
```


```python
len(A.loc[A['cluster']==0])/len(A)
```




    0.8185610010427529




```python
len(A.loc[A['cluster']==1])/len(A)
```




    0.18143899895724713




```python
A1=A.loc[A['cluster']==1]
A0=A.loc[A['cluster']==0]
```


```python
list(df.hashtags.unique())

```




    [nan,
     'AOCspeaks4Me',
     'LastBornsUnite',
     'WritingCommunity',
     'CaptionThis, WritingCommunity',
     'OperationLegend',
     'RIPRobinWilliams, WritingCommunity',
     'BenGarrison',
     'CaptionThis',
     'CaptionThis, WritingCommnunity',
     'TuckerCarlson, WritingCommunity',
     'NoSpoilers',
     'WednesdayWisdom',
     'ThePrestige',
     'Happy4thofJuly, Happy4th',
     'WritingCommunity, CaptionThis',
     'DiaperDon',
     'Batman',
     'StudentsForTrump',
     'KYPrimary',
     'GiveTwoFucks',
     'Karen',
     'McSally, LiberalHack',
     'MacTrump',
     'Hamilton',
     'RIPIanHolm',
     'PrideMonth',
     'BLM, JayPharoah',
     'ConfederateFlag',
     'DDay76',
     'DDay76, DDay',
     'DayMade',
     'ThrowbackThursday',
     'quarantinenewbiemistakes',
     'TheBronx',
     'insomniaproblems',
     'BunkerTrump',
     'angrygetsthingsdone',
     'Grant',
     'MemorialDay',
     'ImWithHer',
     'GodBlessAmerica',
     'EmpireStrikesBack, Empire40th',
     'putthescrewdriverdown',
     'CaptionThis, EmpireStrikesBack, Empire40th',
     'SuperSmashBros',
     'TrumpCommencementSpeech',
     'ObamaWasBetterAtEverything',
     'ff',
     'elvismitchell',
     'frozen',
     'JeffGoldblum',
     'Repost',
     'WritingCommunity, RIPJerryStiller',
     'JaneElliot',
     'MAGA',
     'Morocco2026, Morocco2026, TogetherForOneGoal',
     'RoadTo2026, Morocco, Morocco2026',
     'RoadTo2026, Agadir, Morocco2026',
     'RoadTo2026, Casablanca, Morocco2026',
     'Morocco2026, Morocco, TogetherForOneGoal',
     'Maroc2026, Morocco, Africa, TogetherForOneGoal',
     'TogetherForOneGoal',
     'Morocco2026',
     'Passion, Marrakech, Morocco2026, Morocco2026, TogetherForOneGoal',
     'Morocco2026, TogetherForOneGoal',
     'Russia2018, AtlasLions, Morocco2026',
     'Support, Morocco, Morocco2026',
     'RoadTo2026, Nador, Morocco2026',
     'Morocco2026, football',
     'Support, AtlasLions, TogetherForOneGoal',
     'Support, Morocco, WorldCup, Morocco2026',
     'Russia2018, TogetherForOneGoal, Morocco2026',
     'Support, Morocco2026, AtlasLions, Russia',
     'Soutien, Morocco, AtlasLions, Morocco2026',
     'Maroc2026, Morocco2026, TogetherForOneGoal',
     'Imagine2026, Morocco, Morocco2026',
     'Morocco2026, Morocco, Tangiers, Casablanca, TogetherForOneGoal',
     'Matchday, AtlasLions, Russia2018, Tallinn, Estonia',
     'Morocco2026, Moroccan',
     'Morocco2026, Morocco2026, Moroccan, TogetherForOneGoal',
     'Imagine2026, Morocco2026, TogetherForOneGoal',
     'RoadTo2026, TÃ©touan, Morocco2026',
     'Passion, Marrakesh, Morocco2026, football, Morocco2026',
     'Morocco2026, Morocco, Morocco, TogetherForOneGoal',
     'Marrakesh, Morocco2026, football, Morocco2026',
     'Soutien, Seychelles, Morocco, Morocco2026, TogetherForOneGoal',
     'Imagine2026, Moroccan, Morocco2026',
     'RoadTo2026, Morocco2026, Oujda, TogetherForOneGoal',
     'Morocco2026, AtlasLions',
     'HostCities, MeknÃ¨s, Meknes, Morocco2026',
     'Media, Marrakesh, Casablanca, Morocco2026',
     'Morocco2026, Moroccans',
     'Morocco2026, Mediterranean, Morocco2026, football, TogetherForOneGoal',
     'RoadTo2026, Morocco2026, Marrakesh, TogetherForOneGoal',
     'FanFests, Casablanca, Corniche, AtlanticOcean, WorldCup, FanFest, Morocco2026',
     'Support, Marrakesh, Morocco2026',
     'Morocco2026, Moroccans, football',
     'RoadTo2026, Nas, Meknes, Morocco, Moroccan',
     'Support, Thanks, AtlasLions, Morocco2026',
     'Morocco2026, Morocco',
     'DidYouKnow, Rabat, AtlasLions, Morocco2026',
     'RoadTo2026, Football, Morocco2026, Morocco2026',
     'Support, Morocco2026, TogetherForOneGoal',
     'Morocco2026, Casablanca',
     'RoadTo2026, TogetherForOneGoal',
     'RoadTo2026, Morocco2026, TogetherForOneGoal',
     'Rabat, Morocco2026',
     'Russia2018, AtlasLions, Ukraine, Slovakia, Morocco2026',
     'Ecology, solar, Morocco, hydroelectric, Africa, Morocco2026',
     'MatchDay, AtlasLions, Russia2018, TogetherForOneGoal',
     'Morocco2026, passionate, compact, exciting, TogetherForOneGoal',
     'Tourism, Chefchaouen, Morocco, Morocco2026',
     'RoadTo2026, Casablanca, Marina, airport, TogetherForOneGoal',
     'Stadia, Morocco, Morocco2026',
     'RoadTo2026, Morocco2026, Tangiers, TogetherForOneGoal',
     'Generation2026',
     'RoadTo2026, Morocco2026, Beijing',
     'DidYouKnow, Rabat, Morocco2026, Culture',
     'Morocco2026, Budapest, TogetherForOneGoal',
     'DidYouKnow, Morocco, Morocco2026',
     'DidYouKnow, Marrakesh, Morocco, Morocco2026',
     'Morocco2026, Moroccan, AtlasLions',
     'Morocco2026, Morocco, football',
     'Moroccan, African, Morocco2026, TogetherForOneGoal',
     'RoadTo2026, Morocco2026, compact, innovative, TogetherForOneGoal',
     'RMALIV, UCLFinal, Moroccan, Morocco2026',
     'DidYouKnow, Morocco2026, Morocco2026',
     'Support, Morocco, football, Morocco2026, TogetherForOneGoal',
     'Morocco2026, football, Morocco',
     'AfricaDay2018, Morocco2026, Africa, DimaAfrica, Morocco2026, AfricaDay',
     'DidYouKnow, Morocco, Tangiers, Morocco2026',
     'AtlasLions, Russia2018, Morocco2026, TogetherForOneGoal',
     'RoadTo2026, Morocco2026',
     'RisingStar, Morocco, Russia2018, Morocco2026',
     'RoadTo2026, Morocco2026, Rabat, TogetherForOneGoal',
     'Morocco2026, football, Tangiers',
     'DidYouKnow, Morocco2026',
     'DidYouKnow, Rai, Morocco2026',
     'Morocco2026, Moroccans, AtlasLions, football, Tangiers',
     'Support, Morocco, football, authentic, passion, Morocco2026',
     'HappyBirthday, AtlasLions, Morocco2026',
     'Morocco2026, football, Casablanca, Moroccan, TogetherForOneGoal',
     'Morocco2026, Moroccan, football',
     'AtlasLions, Moroccan, Russia2018, Morocco2026',
     'TogetherForOneGoal, Morocco2026',
     'hostcity, Marrakesh, Morocco2026',
     'DidYouKnow, Football, Athletics, Karate, Handball, Taekwondo, Morocco2026',
     'Legend, Morocco, Morocco2026, TogetherForOneGoal',
     'HostCities, Casablanca, Morocco2026',
     'TogetherForOneGoal, Morocco, Marrakesh, Morocco2026',
     'Passion, Champions, BotolaPro, Morocco',
     'DidYouKnow, Moroccan, Passion, Morocco2026',
     'Moroccan, TropheesUNFP, Morocco2026',
     'HostCities, Oujda, Moroccan, Morocco2026',
     'Stadiums, Oujda, Morocco2026',
     'Congratulations, Tangier, Moroccan, Botola, Morocco2026, Passion',
     'DidYouKnow, Culture, Morocco2026',
     'TogetherForOneGoal, Morocco2026, FootballForAll',
     'DidYouKnow, African, Morocco2026, DimaAfrica, TogetherForOneGoal',
     'DidYouKnow, Moroccan, Morocco2026, Authenticity',
     'HostCities, Rabat, Morocco, African, Morocco2026',
     'HostCities, Rabat, Rabat, Morocco2026',
     'AtlasLions, CoppaItalia, TogetherForOneGoal, Morocco2026',
     'RoadTo2026, Tangiers, Morocco, Morocco2026',
     'HostCities, Tangiers, Tangiers, Morocco2026',
     'Stadiums, Tangiers, Morocco2026',
     'HostCities, Tangiers, Morocco, Morocco, Morocco2026',
     'Marrakesh, Tangiers, Marrakesh, Tangiers, Morocco2026',
     'Marrakesh, Marrakesh, Morocco2026',
     'DidYouKnow, European, African, Morocco2026',
     'GrandStadeDeMarrakesh, Morocco, Morocco2026',
     'StadeElHarti, Marrakesh, StadeElHarti, FIFAClubWorldCup, Morocco2026',
     'StadeElHarti, Marrakesh, Morocco, StadeElHarti, Marrakesh, Morocco2026',
     'LÃ©gende, AtlasLions, Morocco2026',
     'TogetherForOneGoal, Casablanca, Morocco2026',
     'Economy, Marrakech, Casablanca, Morocco2026',
     'HostCities, ElJadida, Morocco2026',
     'Support, Morocco2026',
     'DidYouKnow, AfricanChampionsLeague, Morocco2026',
     'LÃ©gende, Moroccan, Morocco, Spain',
     'TogetherForOneGoal, Morocco2026, Morocco2026',
     'TogetherForOneGoal, Morocco2026, Morocco2026, football, European, Morocco2026',
     'TogetherForOneGoal, DidYouKnow, Morocco2026, Instagram, Moroccan',
     'DimaAfrica, Morocco, Morocco, Africa, Morocco2026',
     'HostCities, ElJadida, Casablanca, Jawhara, Moroccan, music, culture, Morocco2026',
     'RoadTo2026, Stadiums, Morocco, Africa',
     'Support, Myanmar, Morocco2026',
     'RisingStar, RealMadrid, Morocco2026',
     'HostCities, ElJadida, DidYouKnow, Morocco2026, ElJadida, Morocco2026, TogetherForOneGoal',
     'DidYouKnow, Morocco, Morocco, TogetherForOneGoal',
     'DidYouKnow, Rabat, Culture, Morocco2026',
     'AtlasLions, Morocco, TogetherForOneGoal',
     'Support, Morocco, Africa, Senegalese, Morocco2026, TogetherForOneGoal',
     'HappyBirthday, AtlasLions, Morocco, Mexico86, Morocco2026',
     'ICYMI, Morocco2026, Rabat, Morocco2026, TogetherForOneGoal',
     'DidYouKnow, Morocco2026, TogetherForOneGoal',
     'DidYouKnow, AtlasLions, Morocco, Morocco2026, Unity',
     'DidYouKnow, Africa, Morocco2026',
     'HostCities, Nador, DidYouKnow, Morocco2026',
     'Tribute, DimaAfrica',
     'Talent, AFCON, AtlasLions, Russia2018, Morocco2026',
     'DidYouKnow, Morocco, Africa, Morocco, Morocco2026',
     'RoadTo2026, FanFests, Morocco2026, FanFests, Morocco2026',
     'DimaAfrica, Morocco2026, African, Morocco2026, TogetherForOneGoal',
     'Welcome, Morocco, Morocco2026',
     'TropheeHassanII, Rabat, Morocco, Morocco2026',
     'Culture, Rabat, Morocco, Dali, Matisse, Braque, Morocco2026',
     'Tribute, AtlasLions',
     'FGAtWORK',
     'TogetherForOneGoal, Morocco2026, Rabat, Generation2026, Morocco2026',
     'Ambassador, Morocco2026, TogetherForOneGoal',
     'HostCities, Morocco2026, Morocco2026',
     'DidYouKnow, CHAN2018, Morocco, Morocco2026',
     'RisingStar, Moroccan, Generation2026',
     'FIFAVisit2026, PressConference, WorldCup, TogetherForOneGoal',
     'FIFAVisit2026, PressConference, Morocco2026, Casablanca',
     'Hosting, DidYouKnow, Moroccoan, Rabat, Morocco2026, Morocco2026',
     'FIFAVisit2026, Casablanca, Morocco, Marrakesh, Casablanca, Morocco2026',
     'FIFAVisit2026, Casablanca, Casablanca, Authenticity, TogetherForOneGoal',
     'FIFAVisit2026, FanFest, Casablanca, Morocco2026',
     'FIFAVisit2026, Casablanca, TogetherForOneGoal, Authenticity',
     'FIFAVisit2026, Tangiers, Benslimane, TogetherForOneGoal, Morocco2026',
     'FIFAVisit2026, Tangiers, Morocco2026, TogetherForOneGoal',
     'FIFAVisit2026, LGV, Moroccan, LGV, African, Morocco2026, TogetherForOneGoal',
     'FIFAVisit2026, Tangiers, Tangiers, Authenticity, TogetherForOneGoal',
     'FIFAVisit2026, Tangiers, Tangiers, Africa, Europe, Tangiers, Morocco, Authenticity, TogetherForOneGoal',
     'KenyaFootball, Morocco2026, Morocco, African, TogetherForOneGoal',
     'FIFAVisit2026, Tetouan, Agadir, Tetouan, Rif, Authenticity, TogetherForOneGoal',
     'FIFAVisit2026, StadeAdrar, Authenticity, TogetherForOneGoal',
     'FIFAVisit2026, Agadir, DidYouKnow, Agadir, TogetherForOneGoal',
     'FIFAVisit2026, Agadir, Agadir, Authenticity, TogetherForTheOneGoal',
     'FIFAVisit2026, Marrakech, Marrakesh, Morocco, Morocco2026, TogetherForOneGoal',
     'FIFAVisit2026, Authenticity, AtlasLions, SouthKorea, Vietnam, Mexico, Morocco2026, TogetherForOneGoal',
     'FIFAVisit2026, StadeHarti, DidYouKnow, ASM, Heritage, TogetherForOneGoal',
     'FIFAVisit2026, StadeHarti, Marrakesh, Morocco2026, Authenticity, TogetherForOneGoal',
     'FIFAVisit2026, Marrakesh, Morocco2026, Authenticity, TogetherForOneGoal',
     'FIFAVisit2026, Marrakesh, Morocco2026, Marrakesh, TogetherForOneGoal',
     'FIFAVisit2026, HostCities, Morocco2026, Marrakech, Authenticity, TogetherForOneGoal',
     'HappyBirthday',
     'FIFAVisit2026, Morocco, Authenticity, TogetherForOneGoal',
     'RoadTo2026, Unity, Morocco2026, African, Morocco2026, football, TogetherForOneGoal',
     'Ecology, Morocco, Marrakesh, TogetherForOneGoal, Morocco2026',
     'RoadTo2026, Authenticity, Morocco, Morocco2026',
     'MatchDay, Casablanca, Casablanca, Morocco2026',
     'Ambassadors, Unity, Morocco2026, TogetherForOneGoal',
     'DidYouKnow, Morocco, Russia2018, Mali, Gabon, IvoryCoast, AtlasLions, Morocco2026, RoadToRussia',
     'Unity, Morocco2026, Morocco2026, TogetherForOneGoal',
     'FIFARanking, Russia2018, AtlasLion',
     'RoadTo2026, Passion, DidYouKnow, Morocco',
     'ICYMI, Morocco2026, Rabat, inspiring',
     'DidYouKnow, Economy, Morocco2026, TogetherForOneGoal',
     'RoadTo2026, Moroccan, Morocco2026',
     'RisingStar, AtlasLions, Morocco2026',
     'DidYouKnow, Marrakesh, Morocco2026, SportingHeritage',
     'Support, Morocco2026, Morocco',
     'VillesHÃ´tes, Ouarzazate, Morocco2026',
     'Legacy, Morocco2026, HostCities, Morocco2026',
     'FGAtWork',
     'OnThisDay, Morocco, SouthAfrica, Morocco2026',
     'IDSDP2018, Morocco2026, IDSDP, Morocco2026',
     'HostCities, Ouarzazate, Morocco2026',
     'Legend, Brazilian, AtlasLions, Morocco2026',
     'HappyBirthday, AtlasLions, Morocco2026, Unity',
     'RoadTo2026, Port, Morocco2026',
     'RoadTo2026, Unity, Morocco2026, Morocco2026',
     'FIFAWorldCup, DidYouKnow',
     'DidYouKnow, Fez, UNESCO, Morocco, Morocco2026',
     'HostCity, Ouarzazate, Noor, Morocco2026',
     'DidYouKnow, Casablanca, Morocco, Morocco2026, TogetherForOneGoal',
     'DidYouKnow, Morocco, African',
     'Stadium, WorldCup, Africa, Morocco2026',
     'DidYouKnow, Moroccan, African, RoadTo2026, RoadToAfrica, Morocco2026',
     'RisingStar, Tetouan, Moroccan, Morocco2026, AtlasLions',
     'Legend, DidYouKnow, AtlasLions, Morocco2026',
     'DidYouKnow, Tangiers, Morocco, Morocco2026, Africa, TogetherForOneGoal',
     'ATLASLIONS, Morocco2026',
     'Passion, AtlasLions, Russia2018, Morocco2026',
     'Authenticity, Morocco, TogetherForOneGoal',
     'AtlasLions, Morocco2026',
     'DidYouKnow, Moroccan, Morocco2026, TogetherForOneGoal',
     'Rabat, AtlasLions, TogetherForOneGoal',
     'Peru, Iceland, NewJersey, Morocco2026, TogetherForOneGoal',
     'Morocco2026, London, Australia, Colombia, CravenCottage, Morocco2026',
     'Morocco, Uzbekistan, Casablanca, AtlasLions, Russia2018, Morocco2026',
     'Morocco2026, Casablanca, Morocco2026',
     'Casablanca, Casablanca, Morocco2026',
     'Together, Morocco2026, Morocco2026',
     'Morocco, Uzbekistan, AtlasLions, Russia2018, Casablanca, Morocco2026',
     'BidBook, Morocco2026',
     'MarrakechMedina, Morocco2026',
     'HostCities, Marrakech, Marrakech, Morocco, Morocco2026',
     'BidBook, Morocco2026, Morocco2026, BidBook',
     'Marrakech, Morocco2026',
     'HostCities, Agadir, Morocco2026, FanFests, Agadir, Morocco2026',
     'HostCities, Agadir, CHAN2018, FIFA, Morocco2026',
     'SERMAR, AtlasLions, Morocco2026',
     'ICYMI, AtlasLions, Serbia',
     'Matchday, Morocco, Serbia, Turin',
     'RisingStar, AtlasLions, Russia2018, Morocco2026',
     'Morocco, Serbia',
     'Morocco2026, Morocco2026',
     'DidYouKnow, Morocco, Morocco2026, TogetherForOneGoal',
     'Legend, Morocco2026',
     'HostsCities, Agadir, Morocco2026',
     'InternationalDayOfForests, Morocco2026, Azrou, Morocco2026, Morocco2026',
     'Morocco2026, MerciFrancois',
     'InternationalDayofHappiness, DidYouKnow, Marrakech, Morocco2026',
     'Morocco, Morocco2026, TogetherForOneGoal',
     'StadiumsAndCities, Authenticity, Moroccan, Morocco2026',
     'HappyBirthday, AtlasLions, Morocco2026, TogetherForOneGoal',
     'RisingStar, Moroccans, Moroccan, Morocco2026, Generation2026',
     'Casablanca, Flickr, Morocco2026',
     'DidYouKnow, Moroccan, FIFAWorldCup, Morocco2026, Morocco2026',
     'Morocco2026, LIVE',
     'Morocco2026, Casablanca, Morocco2026, TogetherForOneGoal',
     'Morocco2026, FootballForAll, TogetherForOneGoal',
     'Morocco2026, BidBook, English, Arabic, French, Morocco2026',
     'RoadTo2026, Morocco, Morocco, Morocco2026',
     'Morocco, TogetherForOneGoal, Morocco2026',
     'Morocco2026, Morocco, Morocco, Africa, TogetherForOneGoal',
     'DidYouKnow, Morocco, Morocco2026, Generation2026',
     'DidYouKnow, Moroccan, transport, WorldCup, RoadTo2026, Morocco2026',
     'DidYouKnow, Morocco2026, Tourism',
     'OnThisDay, Morocco, Morocco2026, Authenticity',
     'RealMadrid, Morocco, EuropeanFootball, Morocco2026',
     'CHAN2018, AtlasLions, CHANpion, Morocco2026',
     'AtlasLions, Morocco2026, AtlasLions',
     'DidYouKnow, RoadTo2026, Morocco2026',
     'DidYouKnow, AtlasLions, Russia2018, Legend, Morocco2026',
     'DidYouKnow, Morocco, Marrakech, Morocco2026, Authenticity',
     'Morocco2026, Authenticity, Morocco',
     'RisingStar, Generation2026, Morocco2026',
     'RoadTo2026, Authenticity, Morocco2026, football, Morocco2026',
     'DidYouKnow, Morocco, Africa, Sustainability, Morocco2026',
     'RoadTo2026, LGV',
     'Legends, Africanfootball, Berkane, Morocco2026',
     'CAFSymposium, Marrakech, African, Morocco, authentic, Marrakech, Morocco2026, Authenticity',
     'HappyBirthday, Morocco2026, Generation2026',
     'Welcome, AtlasLions, FIFAWorldCup, Morocco2026',
     'DidYouKnow, IWD2018, Morocco, Morocco2026, IWD2018',
     'DidYouKnow, IWD2018, Morocco2026',
     'InternationalWomensDay, African, Morocco2026, IWD2018',
     'Legend, IWD2018, ParalympicGames, Beijing, 100m, 200m, 400m, Morocco2026, InternationalWomensDay',
     'DidYouKnow, IWD2018, Gibraltar, Morocco2026, InternationalWomensDay',
     'InternationalWomensDay, Morocco2026, inclusivity, Morocco2026, IWD2018',
     'Morocco2026, CelebrateAfrica',
     'GoodLuck, AtlasLions, TOTJUV, Morocco2026',
     'discussion, inspiration, action, Marrakech, Morocco2026, FootballForAll',
     'Morocco2026, Africa, sport, football, Morocco, Africa, Morocco2026, CAFSymposium',
     'Morocco, Morocco2026',
     'RoadTo2026, Marrakech, Africa, Morocco2026, FIFAWorldCup, SymposiumCAF, Morocco2026',
     'Olympic, IOC, Morocco2026, Morocco2026',
     'DidYouKnow, Marrakech, Morocco, Morocco2026, Authenticity',
     'FIFAWorldCup, Morocco2026, CelebrateAfrica, FootballForAll',
     'Morocco2026, FootballForAll',
     'Marrakech',
     'CAF, Marrakech, Morocco2026, FootballForAll',
     'DidYouKnow, Ouarzazate, Morocco2026, Authenticity',
     'RoadTo2026, Authenticity, Morocco2026, FootballForAll',
     'DidYouKnow, Morocco, Morocco2026, FIFAWorldCup, Morocco2026, FootballForAll',
     'DidYouKnow, BotolaPro, Morocco2026',
     'RoadTo2026, Tourism, UNESCO, TÃ©touan, Morocco, Morocco2026',
     'DidYouKnow, Futsal, SouthAfrica, Morocco, Morocco2026',
     'Eqality, Inclusion, Morocco2026, FIFA4Equality',
     'DidYouKnow, Moroccan, AtlasLions, CHAN2018, CAFSuperCup, FIFAWorldCup, Morocco',
     'DidYouKnow, JorfLasfar, ElJadida, Africa, FIFAWorldCup, Africa',
     'DIDYOUKNOW, FIFAWorldCup, Morocco',
     'CHANpion, Morocco, CHAN2018, Morocco2026',
     'FollowUs, DIDYOUKNOW, Morocco2026',
     'AtlasLions, Russia, FIFAWorldCup, Morocco, GroupB, Morocco2026, Russia2018',
     'DIDYOUKNOW, Morroco, Morocco2026',
     'DIDYOUKNOW, Africa, TotalCAFSC, WACTPM, Morocco2026',
     'CAFSuperCupFinal, Morocco2026',
     'DIDYOUKNOW, Morocco, Tangiers, KÃ©nitra, Rabat, Casablanca, RoadTo2026, Morocco2026',
     'Morocco2026, FIFAWorldCup, Morocco2026',
     'DIDYOUKNOW, Morocco2026',
     'FootballForAll, Moroccan, Morocco2026, Morocco2026',
     'DIDYOUKNOW, Morocco, Morocco2026, Rabat',
     'SustainableDevelopment, Morocco, Africa, Morocco2026',
     'Moroccan, Morocco2026, Generation2026',
     'Thanks, Moroccan, Morocco, FIFAWorldCup, Morocco2026',
     'DIDYOUKNOW, Marrakech, Morocco, Africa, Morocco2026',
     'DIDYOUKNOW, Morocco, Morocco2026',
     'DIDYOUKNOW, Morocco, FIFA, AtlasLions, Russia2018, FIFAWorldCup, Morocco2026, RoadTo2026',
     'DIDYOUKNOW, Morocco, Mediterranean, Morocco2026',
     'Casablanca, Morocco2026',
     'DIDYOUKNOW, Noor, Morocco2026, GreenEnergy, ClimateChange',
     'Russia, AtlasLions, FIFAWorldCup, Morocco2026, Russia2018',
     'Morocco2026, DimaAfrica',
     'Generation2026, Morocco2026, RoadTo2026',
     'DIDYOUKNOW, Morocco, CHAN, Morocco2026',
     'DIDYOUKNOW, Morocco, RoadTo2026, Morocco2026',
     'DIDYOUKNOW, Noor, Moroccans, Morocco2026',
     'Bostwana, Morocco2026, FIFAWorldCup',
     'DIDYOUKNOW, Morocco, Morocco, Morocco2026, RoadTo2026',
     'DIDYOUKNOW, Marrakesh, COP22, Morocco2026, FIFAWorldCup',
     'Generation2026, Morocco2026',
     'DIDYOUKNOW, Morocco2026, FIFAWorldCup',
     'Moroccan, CasablancaDerby, Morocco2026',
     'CasablancaDerby, CHANpions, Casablanca',
     'BeyondTheGame, FIFAWorldCup, Morocco2026',
     'Morocco, FIFAWorldCup, GroupB, Russia2018, DimaAfrica, Morocco2026',
     'CHANpion, Morocco2026',
     'DIDYOUKNOW, Morocco, sustainability, Morocco, Morocco2026, ClimateChange',
     'DIDYOUKNOW, FIFA, Marrakech, Morocco, Morocco2026',
     'CHANpion, ElYamiq, Morocco2026, DimaAfrica',
     'CHAN2018, Morocco, CHAN2018, ReadyFor2026, Morocco2026',
     'FIFAWorldCup',
     'DIDYOUKNOW, FIFAWorldCup, Morocco2026',
     'RoadTo2026, Tangiers, Morocco2026',
     'DimaAfrica, CHAN2018, RoadTo2026, Morocco2026',
     'CHAN2018',
     'AyoubElKaabi, Morocco2026, DimaAfrica',
     'CHAN2018, Morocco2026, DimaAfrica',
     'AtlasLions, TotalCHAN2018, Morocco2026, DimaAfrica',
     'Morocco, TotalCHAN2018, Roadto2026, Morocco2026',
     'Morocco, Africa, Morocco2026',
     'AKSC, StaySafe',
     'AKSC',
     'AKSC, StaySafe, KOTBEC',
     'StaySafe, AKSC',
     'AKSC, StaySafe, Zumzum',
     'AKSC, StaySafe, Believe',
     'AKSC, LetsFightCovid19',
     'AKSC, Family',
     'LetsFightCovid19, AKSC',
     'PSMTAtKashimbilla',
     'AKSC, LetsFightCovid19, KanoPillarsvsKotoko',
     'AKSC, LetsFightCovid',
     'AKSC, LetsFightCovid19, MondayMotivaton',
     'StaySafe, AKSC, LetsFightCovid19',
     'AKSC, LetsFightCoVid19',
     'AKSC, LetsFightCovid19, Family',
     'AKSC, WashYourHands, Sanitise, MaintainSocialDistancing',
     'FathersDay, AKSC, LetsFightCovid19, Family',
     'ApapaFlagOff',
     'FamilyContest, AKSC, LetsFightCovid19',
     'AKSC, AKSC, LetsFightCovid19',
     'StateOfPlay',
     'StateOfThePlay',
     'AKSC, LetsFightCovid19, MondayMotivation',
     'AKSC, KOTNKA, LetsFightCovid19',
     'MatchRewind, DREKOT',
     'AKSC, DREKOT',
     'AKSC, LetsStaySafe',
     'AKSC, ThisTooShallPass',
     'AKSC, Moments, LetsFightCovid19',
     'AKSC, EidMubarak, LetsFightCovid19',
     'AKSC, NKAKOT, LetsFightCovid19',
     'AKSC, staysafe',
     'AKSC, Fabulous4Life, Family',
     'AKSC, AKSC, FabulousWomen',
     'edtech, remotelearning, distancelearning, bettertogether',
     'AKSC, CaptainFantastic, ThrowbackThursday',
     'AKSC, InternationalNursesDay2020',
     'MothersDay, AKSC, LetsFightCovid19',
     'distancelearning, hybridlearning, CARESAct, edmodocares, education',
     'gamification, edtech, gamify',
     'edtech, bettertogether, edmodochat, distancelearning, remotelearning',
     'AKSC, StaySafe, LetsFightCovid19',
     'MondayMotivation, ThisTooShallPass, AKSC, letsfightcovid19',
     'MondayMotivation, AKSC, LetsFightCovid19',
     'Family, LetsFightCovid19',
     'egypt, bettertogether, edtech, distancelearning, remotelearning',
     'edtech, PD, edmodochat',
     'AKSC, StayatHome',
     'AKSC, StayHome',
     'AKSC, StayAtHome, LetsFightCovid19',
     'ProjectInProgress, AKSC, StayAtHome',
     'AKSC, StayHome, LetsFightCovid19',
     'AKSC, StayAtHome',
     'StayAtHome, AKSC, LetsFightCovid19',
     'ThrowbackThursday, AKSC',
     'AKSC, LetsFightCovid19, StayAtHome',
     'distancelearning, remotelearning, EdmodoStory, sel, edtech, bettertogether',
     'bettertogether',
     'distancelearning, remotelearning, edtech',
     '14daylockdown, AKSC, StayAtHome',
     'distancelearning, remotelearning, edmodochat, edtech, bettertogether',
     'StayHome, AKSC, LetsFightCovid19',
     'AKSC, RevivalOfHope, StayAtHome',
     'FootballClubChallenge',
     'StayHome, AKSC, LetStaySafe, LetsFightCoronaTogether',
     'AKSC, LetStaySafe, StayHomeSaveLives',
     'stayhomechallenge',
     'distancelearning, remotelearning, bettertogether',
     'AKSC, LetStaySafe',
     'AKSC, LetStaySafe, LetsFightCovid19together',
     'AKSC, LetStaySave',
     'stayAtHomeChallenhge',
     'AKSC, stayhomechallange',
     'AKSC, KOTBEC',
     'GPLGoalChain, AKSC',
     'StayAtHomeChallange, AKSC',
     'bettertogether, remotelearning, distancelearning',
     'COVID19, COVID2019, AKSC',
     'COVID19, AKSC',
     'ALLKOT, AKSC',
     'BetterTogether',
     'AKSC, ALLKOT',
     'AKSC, KOTSHA',
     'KOTSHA',
     'KOTSHA, AKSC, KOTSHA',
     'NanaAmaChallenge, AKSC, KOTSHA',
     'WePromise, ItWillBeOK',
     'WePromise, WeRelyOnYou',
     'WePromise',
     'KOTOKOSHARKS, AKSC, KOTSHA',
     'AKSC, KORSHA',
     'DistanceLearning, remotelearning, edtech, SAMR',
     'InternationalWomensDay',
     'fun',
     'AKSC, Family, KOTSHA',
     'dontpanic, breathe, itwillbeok, WePromise',
     'KARKOT',
     'AKSC, KARKOT',
     'Karela, AKSC, KARKOT',
     'BetterTogether, remotelearning, distancelearning',
     'BetterTogether, remotelearning, DistanceLearning',
     'bettertogether, remotelearning, DistanceLearning',
     'AFCONQ, AKSC, KARKOT',
     'KOTBEC',
     'BetterTogether, remotelearning, distancelearning, edtech, edmodochat, geniallychat, formativechat',
     'BetterTogether, Edmodo, edtech, distancelearning, remotelearning, edtech, edmodochat, geniallychat, formativechat',
     'Edmodo, DistanceLearning, RemoteLearning, edtech, edmodochat',
     'AKSC, AKSC',
     'AKSC, KOOTBEC',
     'AKSC, HEAKOT',
     'smart, slowchat, edmodochat, geniallychat',
     'slowchat, tools, educators, teamwork, edmodochat, geniallychat, edtech',
     'HappyNotPerfect, Edmodo, game, daily, mindfulness, routine, SEL, edtech, gamification, free',
     'slowchat, collaboration, classroom, professionaldevelopment, edmodochat, geniallychat, edtech',
     'edmodochat',
     'slowchat, educators, engage, students, classroom, edmodochat, geniallychat, edtech',
     'edmodolove',
     'slowchat, team, community, edmodochat, geniallychat, edtech',
     'geniallychat, edmodochat',
     'slowchat, team, collaborates, edmodochat, geniallychat, edtech, teamwork',
     'NFL, WashingtonFootballTeam',
     'slowchat, Engagement, Collaboration, TeamWork, edmodochat, geniallychat, edtech, BetterTogether',
     'BetterTogether, EdmodoChat, Edmodo, EdmodoLove, EducatorsMakeTheWorldaBetterPlace, StudentsEnlivenOurWorld, ValentinesDay',
     'BRAVO, iste, edmodochat, DigCitCommit, digitalcitizenship',
     'Update, Edmodo, edmodochat, edtech, WhatsNew',
     'ISTE, DigCitCommit, edmodochat',
     'edtech, DigCitCommit, ISTE, Edmodochat, digitalcitizenship, edchat',
     'DigCitCommit, iste',
     'Edmodo, edtech, DigCitCommit, iste',
     'DigitalCitizenship, skill, students, leaders, edtech, ISTE2020, ISTE, DigCit, EdmodoChat',
     'BetterTogether, YouGotThis, MondayMotivation, MondayThoughts, MondayMood, MondayVibes, EdmodoChat',
     'FREE, DigCit, DigCitCommit, DigitalCitizenship, digital, engagement, edtech, professionaldevelopment, PD, edmodochat',
     'ThrowbackThursday, edtech, edmodochat, MoOurMascot, Memories',
     'ThrowbackThursday, opportunities, learning, Digital, EdmodoCon, edtech, edmodochat',
     'WellnessWednesday, Mindfulness, Meditation, BETT, HNP, SEL, mindfulness, wellness, education, edmodochat',
     'edtech, collaboration, connections, bettertogether, education, edmodochat',
     'edtech, edmodochat, DigitalLearning, blendedlearning, flippedlearning',
     'Global, Collaboration, Project, Edmodo, PBL, globalcollaboration',
     'ThankfulThursday, gratitude, edtech, SEL, edmodochat',
     'Designers, researchers, education, teamwork, community, design, edmodochat, edtech',
     'Update, edmodochat, edtech',
     'Edmodo, edmodochat, edtech',
     'Quiz, Teachers, edtech, edmodochat',
     'bett2020',
     'flexible, learning, community, bett2020, edtech, edmodochat, edmodolove, alwayslearning, bettertogether',
     'bett2020, edtech, alwayslearning, bettertogether',
     'sel, edtech, bett2020',
     'bett2020, edmodochat, edtech',
     'ComebackSzn',
     'edtech, free, alwayslearning, bettertogether, Bett2020',
     'demoswap, bett2020',
     'bett2020, demoswap, edtech, alwayslearning, bettertogether',
     'bett2020, demoswap',
     'demoswap, bett2020, bettertogether, edtech',
     'demoswap, Bett2020',
     'Bett2020, alwaysteaching, alwayslearning, bettertogether, edtech, studentcentered',
     'MotivationMonday, fitnessmotivation',
     'bett2020, gamification, edtech',
     'bett2020, edtech, edmodochat, edmodolove',
     'sel, edtech, bett2020, edmodochat, edmodo',
     'Edmodo, SEL, bett2020, edtech, edmodochat, edmodolove',
     'bett2020, blendedlearning, flippedlearning, SEL, gamification',
     'Bett2020',
     'Bett2020, edmodochat, edtech',
     'SEL, bett2020, edmodochat, edtech, digitallearning, blendedlearning, flippedlearning',
     'bett2020, edtech, alwayslearning, bettertogether, flippedlearning, blendedlearning, digitalclassroom, digitalcitizenship',
     'edmodochat, Bett2020',
     'livingtolearn, learningtolive, Edmodo, edmodochat, alwayslearning',
     'bett2020, edtech, alwayslearning, bettertogether, GoTeam',
     'bett2020, computerscience, edmodochat, edmodolove, edtech, alwayslearning, bettertogether',
     'Edmodo, Bett2020',
     'bett2020, edtech, edmodochat',
     'Edmodo, bett2020, edmodochat, edtech',
     'bett2020, edmodochat',
     'Edmodo, flippedlearning, blendedlearning, SEL, bett2020, edtech, alwayslearning',
     'dailymotivation',
     'Edmodo, bett2020, edmodochat, edtech, alwayslearning, bettertogether',
     'excel, edtech, bett2020',
     'inspiring, inspiringview',
     'Edmodo, edmodochat, edtech, SEL',
     'EdmodoLove, edmodochat, flipclasschat, bettertogether',
     'MondayMotivation, edmodochat',
     'edmodochat, WordsOfWisdom',
     'edmodochat, edtech',
     'Bett2020, edmodochat, edchat, edtech',
     'BlackLivesMatter',
     'FollowFriday, ff, edmodochat, edtech, ConnectED, bettertogether',
     'SEL, Edmodo, edmodochat, happiness, edchat',
     'cycling, peloton',
     'Edmodo, edmodochat, collaboration, elemchat',
     '49ersfaithful, 49ers, Charity, CommunityService',
     'Edmodo, teaching, students, edmodochat, edtech, SEL',
     'quoteoftheday, MotivationalMonday, teaching, edmodochat, edtech',
     'Edmodo, Try, computerscience, teaching, edmodochat, edmodo, NewYear',
     'Edmodo, mindful, hnp, sel, happiness, edmodochat',
     'slowchat, newyear, backtoschool, edmodochat',
     'Apple, Apps, edmodochat, iOs',
     'Android, Apps, edmodochat',
     'razorep',
     'RAZOREP',
     'SizeDoesMatter',
     'Equality',
     'RayshardBrooks',
     'BLM',
     'TBI',
     'JusticeforBreonnaTaylor, JusticeForGeorgeFloyd, JusticeForAhmaudArbery',
     'MonsterEnergy',
     'SATISFIED',
     'SeanSuiter',
     'JusticeForGeorgeFloyd',
     'EndRacism, Equality, freedom, LoveOverHate',
     'phynofest',
     'monsterenergy',
     'TrayvonMartin',
     'TraceLivePhyno, DealWithIt',
     'JusticeForAll',
     'FightingForJustice',
     'JusticeForAllOfThem',
     'GeorgeFloyd, PhilandoCastille, FreddieGray, TrayvonMartin, TamirRice, MikeBrown',
     'GeorgeFloyd, SanJose, EnoughIsEnough, WeAreTired',
     'PHM',
     'DerekChauvin',
     'DerekChauvin, MohamedNoor',
     'warlords',
     'GeorgeLloyd, DerekChauvin',
     'BIKO',
     'GeorgeFloyd',
     'f, PHM',
     'gettheinfo',
     'Police',
     'GETTHEINFO',
     'phynoxnewage',
     'PhynoXNewAge',
     'TheLastDance',
     'Gonzalo, Hanna',
     'FridayMotivation, coronavirus',
     'nfl',
     'coronavirus, SaveOurEducation',
     'COVID19, InvestInKids',
     'Lebanon',
     'RandomThoughts',
     'COVID19',
     'BacktheFrontline',
     'coronavirus, TuesdayMotivation',
     'coronavirus',
     'HurricaneSeason',
     'MandelaDay',
     'WorldEmojiDay',
     'MomentofLift',
     'Yemen, coronavirus',
     '100DaysofReading',
     'DonateCrypto, Bitcoin, CryptoTaxTips',
     'FathersDay',
     'YemenCantWait',
     'Yemen',
     'EqualityCantWait',
     'BreonnaTaylor, SayHerName',
     'COVID19, Syria',
     'ThursdayThoughts',
     'NationalVideoGameDay',
     'BookTube',
     'TuesdayMotivation',
     'FourthofJuly',
     'WhyISponsor',
     'NationalSuperheroDay',
     'WeGotThisSeattle',
     'COVID19, solidaritypledge',
     'WorldHealthDay',
     'COVID19, SafeHands',
     'SyriaConf2020',
     'TuesdayThoughts',
     'EqualityCantWait, GenerationEquality',
     'GenerationEquality, EqualityCantWait, IWD2020',
     'IWD2020, InternationalWomensDay',
     'Juneteenth',
     'Syria, coronavirus',
     'NASA',
     'tbt',
     'generationequality',
     'MomentofLiftBookClub, Sweepstakes',
     'GETCities',
     'MLKDay',
     'MomentofLiftBookClub, sweepstakes',
     'MomentofLiftBookClub',
     'MomentofLift, BookTube',
     'EndAIDS',
     'paidleave',
     'DAC2020, ProtectAGeneration',
     'UHCDay',
     'GraceHopper, WomeninSTEM, CSEdWeek',
     'GivingTuesdayKids',
     'LatinaEqualPay, EqualityCantWait',
     'NationalHikingDay',
     'ICPD25',
     'MomentofLift, GoodreadsChoice',
     'NationalSTEMDay',
     'RebootRepresentation',
     'HappyHalloween',
     'WisdomfromtheTop',
     'EndPolio',
     'PaidFamilyLeave',
     'coronavirus, NoChildLaborDay',
     'AllWomanSpacewalk, EqualityCantWait',
     'YaPasarÃ¡, COVID19',
     'FemaleFoundersComp',
     'IDELA, ECD, IDELA',
     'Goalkeepers19',
     'WiW2019, EqualityCantWait',
     'DayoftheGirl, RiseUpTogether',
     'Period',
     'COVID19, VaccinesWork',
     'Goalkeepers2019',
     'HealthForAll',
     'UNGA',
     'GenerationEquality',
     'BlackoutTuesday',
     'HealthForAll, HLMUHC',
     'InternationalChildrensDay',
     'InsideBillsBrain',
     'coronavirus, EndChildMarriage',
     'NationalSmileDay, WhyISponsor',
     'BeyondSelfCare',
     'GetOutTheBias',
     'ThankAHungerHero',
     'BeKind21',
     'ThankyouAnganwadiDidi',
     'COVID19Pandemic, TuesdayMotivation',
     'WomensEqualityDay, EqualityCantWait',
     'BlackWomensEqualPayDay, EqualityCantWait',
     'NosesOn',
     'MomentofLift, EqualityCantWait',
     'BookLoversDay',
     'breastfeedinghero',
     'ProtectAGeneration',
     'WorldBreastfeedingWeek, breastfeedinghero',
     'AI',
     'apollo50',
     'Coronavirus, MondayMotivation',
     'G7',
     'Somalia, coronavirus',
     'SAVEWITHSPORTS',
     'Partner4Children, COVID19',
     'MalalaDay',
     'IDELA, IDELA',
     'USWNT',
     '4thofJuly, TBT',
     'Mozambique, coronavirus',
     'InternationalNursesDay',
     'FlashbackFriday',
     'MondayMotivation',
     'MothersDay',
     'equalitycantwait',
     'Goalkeepers2018, FreePeriods, NationalSelfieDay, FBF',
     'MothersDay, ThankYouMom',
     'HungerChallenge, COVID19',
     'FridayFeeling',
     'PositiveDisruption, PathwaysCommission',
     'LoveTakesAction',
     'BestFriendsDay',
     'Covid_19',
     'WD2019',
     'IndianElections2019, EqualityCantWait',
     'Gender7, G7France',
     'SAVEWITHSTORIES',
     'ThePowerOf',
     'C19ImpactInitiative, GivingTuesdayNow, coronavirus',
     'Yemen, Coronavirus',
     'unspokenstories',
     'MyNextGuest',
     'Sesame50',
     'GivingTuesdayNow',
     'AmericasFoodFund, COVID19',
     'coronavirus, GivingTuesdayNow',
     'RedNoseDay',
     'HurricanePrep, HurricaneStrong',
     'ButThatsAnotherStory',
     'AnswerTheCall',
     'TheNextHighFive',
     'ThankATeacher, TeacherAppreciationDay',
     'BNPodcast, MomentofLift',
     'VaccinesWork, WorldImmunizationWeek',
     'Somalia, VaccinesWorkForAll',
     'SuperSoulConversations',
     'independentbookstoreday',
     'SuperSoulConversations, MomentofLift',
     'VaccinesWork',
     'TIME100',
     'CantJustPreach',
     'WorldImmunizationWeek',
     'coronavirus, Ramadan',
     'ForumonLeadership',
     'TIMESUP',
     'WITW',
     'NationalLibraryWeek',
     'NationalLibraryWeek, LibrariesTransform',
     'freeperiods',
     'WomensHistoryMonth',
     'womenintech, diversityintech',
     'womenshistorymonth',
     'heroesinthefield',
     'EarthDay',
     'girlsinSTEM',
     'Give4CovidRelief',
     'Web30, ForTheWeb',
     'IfThenSheCan',
     'SheInspiresMe',
     'IWD2019',
     'TBT',
     'ThankfulThursday',
     'AWSportsSummit2020, ShiftingMindsets',
     'TakeYourPlace, AWSportsSummit2020',
     'AWSportsSummit2020',
     'AWSportsSummit, TakeYourPlace',
     'AWSportsSummit',
     'SportsPanorama, AWSportsSummit',
     'AWSportsSummit, SportsPanorama',
     'holidays',
     'ransomware',
     'SheTooCan, AWSportsSummit',
     'NYIF',
     'IoTinActionMS, IoT',
     'CSEdWeek',
     'UNIDPD',
     'ByTheNumbers',
     'HoloLens2',
     'FarmBeats',
     'CyberMonday',
     'PMBinMali',
     'BlackFriday',
     'NewNormal',
     'AllianzMicrosoft',
     'Nigeria',
     'Azure',
     'X019',
     'MSIgnite',
     'LifeIsShort',
     'HalaMadrid',
     'PAGMI',
     'ML',
     'AzureAI',
     'MSPowerBI',
     'TikTok',
     'MSFTEduChat',
     'IoT, IoTinActionMS',
     'RNASwift',
     'Teachers',
     'SAPTechEd',
     'MSDyn365',
     'SIEM',
     'naturaldisasters',
     'AI, MicrosoftTeams',
     'TakeResponsibility',
     'Gears5',
     'IBC2019',
     'ImagineCup',
     'MinecraftEdu',
     'IoT, Microsoft, MSPartner, IoTinActionMS',
     'AsoVillaToday',
     'TakeResponsibility, WearAMask',
     'COVID19Nigeria',
     'AKKProject',
     'Nigeria, Abuja, Kaduna, Kano, AKKProject',
     'PTFCOVID19',
     'Buhari',
     'APC',
     'TakeResponsibility, COVID19Nigeria',
     'StaySafe',
     'DrugHelpNet, WDR2020',
     'FactsForSolidarity, WDR2020',
     'SayNoToRapist, Hausa',
     'FoodSecurity, EndHunger',
     'CHEMCI',
     'RIJF',
     'COVID19Nigeria, StaySafe',
     'WWDC2020',
     'WearAMask, MondayMotivation',
     'WorldSickleCellDay, KnowYourGenotype, SickleCellAwareness',
     'WAAW2019',
     'AMRactionNG, NAAW2019',
     'WorldFoodDay',
     'WordFoodDay, ZeroHunger',
     'ZeroHunger, WorldFoodDay19',
     'ZeroHunger',
     'AgricExpo2019',
     'EidMubarak',
     'ZeroRejectNg',
     'ZeroRejectNg, FoodSafety',
     'APPEALSng',
     'GovtAtWorkNG',
     'Bee',
     'Aflatoxins',
     'Aflasafe',
     'NCARD19',
     'HappyEaster',
     'seedconnect2019',
     'SeedConnect',
     'SeedConnect2019',
     'Nigeria, Netherlands',
     'WomensDay, women, BalanceforBetter',
     'OIIEwebinarseries, OIIEchat',
     'covid19',
     'workfromhome, covid19',
     'OIIEchat',
     'DemoDay, Nigeria, DigitalNigeria, DemoDay',
     'NigeriaCOVID, DemoDay',
     'NigeriaCOVID19innovationchallenge',
     'NigeriaCOVID',
     'GrowNigeria, EatLocal',
     'EatNigerian, BuyNigerian',
     'SUCKEBBI2020',
     'COVID19, NITDA4COVID19',
     'SUCABUJA2020',
     'DigitalEconomy, stayConnected, OIIETweetChat',
     'SUCABUJADAY2',
     'COVID19InnovationChallenge, OIIEchat',
     'COVID19Pandemic, tech4COVID19, DigitalEconomy, stayConnected, OIIETweetChat, NITDATech4COVID19',
     'staysafe, stopDspread, takeresponsibility',
     'covid, staysafe, takeresponsibility',
     'TBT, GITEX2019',
     'TBT, GITEXbootcamp2019',
     'COVID19NIGERIA',
     'Dr, SUCJIGAWA2020',
     'SUCJIGAWA2020',
     'SUCJIGAWADAY4',
     'SUCJIGAWADAY3, CustomerAcquisition',
     'SUCJIGAWADay2',
     'SUCJIGAWADAY3',
     'SUCJIGAWADAY2, CustomerValidation, SUC2020JIGAWA',
     'DabaraTech, FutureHack',
     'CodeSpace, FutureHack',
     'localcontentglobalsolutions, innovation, digitalEconomy',
     'FutureHack2020',
     'FutureHack',
     'Hackathon',
     'FutureHack2002',
     'Futurehack',
     'OIIEChat',
     'oiiechat',
     'DigitalNigeria',
     'OIIE',
     '2020NewYear',
     'FederalUniversityofAkure, AhmaduBelloUniversityZaria, TechXhub, Roarhub, NileUniversity',
     'enigeria, OIIE',
     'Cisco',
     'googledigitalskill, hapticsng, enigeria, OIIEchat',
     'enigeria, OIIEchat',
     'OIIE, enigeria, OIIEchat',
     'Startup, enigeria, OIIEchat',
     'Startup',
     'GITEX2019',
     'startup, enigeria, OIIEChat',
     'enigeroa, OIIEchat',
     'eNigeria2019',
     'FMoCDENigeria, eNigeria2019',
     'eNigeria2019, MyeNigeriaExperience',
     'eNigeria2019, NITDA, innovatetoTransform, startups, ngrinnovation',
     'eNigeria2019, PMBAteNigeria, MyeNigeriaExperience',
     'eNigeria2019, PMBAteNigeria, MyeNigeriaExperience, OIIE',
     'startupseries',
     'startupseries, NITDAsponsoredstartups, Nigeriastartups, GITEX2019',
     'startupseries, GITEX2019',
     'startupseries, NITDAsponsoredstartup',
     'startupseries, NITDAsponsoredstartups, Nigeriastartup',
     'startupseries, NITDAsponsoredstartups, Nigeria',
     'startupseries, NITDAsponsoredstartups, Nigeriastartups',
     'GITEX2019, startupseries, NITDAsponsoredstartups, Nigeriastartups, GITEX2019',
     'GITEX',
     'GITEX2019, techisthefuture, GITEX2019, OIIE, NITDA',
     'InPhotos, NITDA, GITEX2019, techisthefuture, GITEX2019, OIIE, NITDA',
     'DGNITDA, GITEX2019',
     'NITDA',
     'NITDA, GITEX, techisthefuture, GITEX2019, OIIE, NITDA',
     'techisthefuture, GITEX2019, OIIE, NITDA',
     'InPhotos, GitexTechWeek2019, techisthefuture, GITEX2019, OIIE, NITDA',
     'DGNITDA, techisthefuture, GITEX2019, OIIE, NITDA',
     'NITDA, GITEX2019, OIIE',
     'techisthefuture, OIIE, NITDA19, GITEX2019',
     'GITEX2019, Techisthefuture',
     'GITEX2019, Techisthefuture, GITEX2019, NITDA, OIIE',
     'OIIE, Nigeria, Nigeria',
     'GITEX2019, NITDAsponsoredstartups, Bootcamp, OIIE',
     ...]




```python
#df.loc[df['hashtags']==re'(?:https?:\/\/)?(?:www\.)COVID']
100*(len(df[df['hashtags'].str.contains('(Vaccines|StayHome|stayhome|Covid|LetStaySafe|COVID|WearAMask)', regex=True,na=False)][['user','hashtags']])/len(df))
```

    C:\Users\Hp\Anaconda3\lib\site-packages\ipykernel_launcher.py:2: UserWarning: This pattern has match groups. To actually get the groups, use str.extract.
      
    




    5.058327640894481




```python
df_covid=df[df['hashtags'].str.contains('(Vaccines|StayHome|stayhome|Covid|LetStaySafe|COVID|WearAMask)', regex=True,na=False)][['user','hashtags']]

```

    C:\Users\Hp\Anaconda3\lib\site-packages\ipykernel_launcher.py:1: UserWarning: This pattern has match groups. To actually get the groups, use str.extract.
      """Entry point for launching an IPython kernel.
    


```python
len(df_covid['user']==A0[['user']])/len(df)
```




    0.002078402092699381




```python
100*(len(df[df['hashtags'].str.contains('(Nigeria|naija|9ja)', regex=True,na=False)][['user','hashtags']])/len(df))
```

    C:\Users\Hp\Anaconda3\lib\site-packages\ipykernel_launcher.py:1: UserWarning: This pattern has match groups. To actually get the groups, use str.extract.
      """Entry point for launching an IPython kernel.
    




    1.279872065746345


