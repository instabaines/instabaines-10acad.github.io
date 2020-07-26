---
title: "Twitter Influencers [Digital Marketing]"
date: 2020-07-26
tags: [data wrangling, data science, messy data, Twitter crawling, Digital Marketing]
header:
  image: "/images/perceptron/Twitter-Logo.jpg"
excerpt: "Data Wrangling, Data Science, Messy Data,Twitter,Data Minining"
mathjax: "true"
---

<big>African Influencer for twitter marketing
</big>

<h1>Import the necessary libraries</h1>


```python
pip install textblob
```


```python
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
```


```python
from requests import get
from requests.exceptions import RequestException
from contextlib import closing
from bs4 import BeautifulSoup
import pandas as pd
import os, sys
import re
import fire
from collections import Counter
```

The crawler below fetch data from the internet, this is iur first source of information


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
res = get_elements('https://africafreak.com/100-most-influential-twitter-users-in-africa')
res
```


```python
res[0]
```


```python
mylist_100_inf=[]
for i in res:
    if len(mylist_100_inf)<100:
        mylist_100_inf.append(i.split('@')[-1].split(')')[0])

```


```python
mylist_100_df=pd.DataFrame(mylist_100_inf)
#mylist.to_csv('10 influencers.csv')
```


```python
url= 'https://www.atlanticcouncil.org/blogs/africasource/african-leaders-respond-to-coronavirus-on-twitter/#east-africa'
response = simple_get(url)
```


```python
response
```


```python
res = get_elements(response, search={'find_all':{'class_':'wp-block-embed__wrapper'}})
res
```


```python

```


```python
regex ='#https?://twitter\.com/(?:\#!/)?(\w+)/status(es)?/(\d+)#is'
a=re.match('((https?://)?(www\.)?twitter\.com/)(@|#!/)?([A-Za-z0-9_]{1,15})(/([-a-z]{1,20}))?','https://twitter.com/SE_Rajoelina/status/1241101811647500288')
#print(a.group(0))

```


```python

```


```python
mylist=[]
for i in res:
    a=re.search('((https?://)?(www\.)?twitter\.com/)?(@|#!/)([A-Za-z0-9_]{1,15})(/([-a-z]{1,20}))?',str(i))
    if a!=None:
        mylist.append((a[0]))
mylist[:10]
#mylist=pd.DataFrame(mylist[:10])
#mylist.to_csv('10 African leaders.csv')
```


```python
consumer_key = '#########'
consumer_secret = '#########'
access_token = '#########'
access_token_secret = '#########'
```


```python
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

    
```


```python

```


```python
##No of likes
fav_count={i:'x' for i in mylist}
No_of_followers={i :'x'for i in mylist}
No_of_following={i :'x'for i in mylist}
for i in mylist:
        user = api.get_user(i)
        fav_count[i]=user.favourites_count
        No_of_followers[i]=user.followers_count
        No_of_following[i]=user.friends_count
        
```


```python
#for tweet in tweepy.Cursor(api.user_timeline,id=mylist[0]).items():
   # print (tweet)
    #op=tweet._json
#op['entities']['hashtags'][0]['text']
op
```


```python
##Mentions
mention_count={i:'x' for i in mylist}
for i in my_list:
    for results in tweepy.Cursor(twitter_api.search, q=i).items(200):
        op=tweet._json  
        count=+1
        mention_count[i]=count
    
```

<b>Twitter cawler use twitter API to fetch required information</b>


```python
for j in mylist_100_inf:
    try:
        tweets = api.user_timeline(screen_name=j, 
                           # 200 is the maximum allowed count
                           count=200,
                           include_rts = False,
                           # Necessary to keep full_text 
                           # otherwise only the first 140 words are extracted
                           tweet_mode = 'extended')
    except tweepy.TweepError:
            continue
    for tweet in tweets:
            tweet_json=tweet._json
            date=[]
            tweet=[]
            users_mention=[]
            hashtag=[]
            retweet_count=[]
            retweeted=[]
            location=[]
            followers=[]
            verified=[]
            country=[]
            fav_count=[]
            user=[]
            comment=[]
            following=[]

            
            i=tweet_json
            #print (j)
            #country.append(i['place']['country'])
            fav_count.append(i['favorite_count'])
            user.append(j)
            verified.append(i['user']['verified'])
            followers.append(i['user']['followers_count'])
            location.append(i['user']['location'])
            date.append(i['created_at'])
            tweet.append(i['full_text'])
            try:
                users_mention.append(i['entities']['user_mentions'][0]['screen_name'])
            except IndexError as e:
                users_mention.append('Nan')
            try:
                hashtag.append(i['entities']['hashtags'][0]['text'])
            except IndexError as e:
                hashtag.append('Nan')
            retweet_count.append(i['retweet_count'])
            retweeted.append(i['retweeted'])
            if i['in_reply_to_status_id']:
                comment.append(1)
            else:
                comment.append(0)
            following.append(i['user']['friends_count'])
            dict={'date_created':date,'user':user,'tweet':tweet,'user_mention':users_mention,'retweet_count':retweet_count
                ,'retweeted':retweeted,'location':location,'followers':followers,'following':following,'verfied':verified,'hashtag':hashtag,'comment':comment,'likes':fav_count}
            df=pd.DataFrame(dict)
            try:
                I_A=pd.concat([df,I_A])
            except NameError:
                I_A=df
    
  
    

```


```python

I_A.to_csv('Influencer.csv')
```


```python

Ireachscore=[]
Ipopularityscore=[]
for i in mylist_100_inf:
    retweet=I_A.loc[I_A['user']==i, 'retweet_count'].sum()
    Ipopularityscore.append(retweet)
    try:
        Ireachscore.append(I_A.loc[I_A['user']==i, 'followers'].unique().item()-I_A.loc[I_A['user']==i, 'following'].unique().item())
    except ValueError  :
        Ireachscore.append(0)
    
```


```python

Areachscore=[]
Apopularityscore=[]
for i in mylist:
    retweet=f_A.loc[f_A['user']==i, 'retweet_count'].sum()
    Apopularityscore.append(retweet)
    try:
        Areachscore.append(f_A.loc[f_A['user']==i, 'followers'].unique().item()-f_A.loc[f_A['user']==i, 'following'].unique().item())
    except ValueError  :
        Areachscore.append(0)
    
```


```python

```


```python
influencer_score=pd.DataFrame({'Influencer':mylist_100_inf,'reach':Ireachscore,'Popularity':Ipopularityscore})
Leaders_score=pd.DataFrame({'Influencer':mylist,'reach':Areachscore,'Popularity':Apopularityscore})
influencer_score.to_csv('influencer_score.csv')
Leaders_score.to_csv('Leaders_score')
Leaders_score

```


```python

for j in mylist:
    try:
        tweets = api.user_timeline(screen_name=j, 
                           # 200 is the maximum allowed count
                           count=200,
                           include_rts = False,
                           # Necessary to keep full_text 
                           # otherwise only the first 140 words are extracted
                           tweet_mode = 'extended')
    except tweepy.TweepError:
            continue
    for tweet in tweets:
            tweet_json=tweet._json
            date=[]
            tweet=[]
            users_mention=[]
            hashtag=[]
            retweet_count=[]
            retweeted=[]
            location=[]
            followers=[]
            verified=[]
            country=[]
            fav_count=[]
            user=[]
            comment=[]
            following=[]

            
            i=tweet_json
            #print (j)
            #country.append(i['place']['country'])
            fav_count.append(i['favorite_count'])
            user.append(j)
            verified.append(i['user']['verified'])
            followers.append(i['user']['followers_count'])
            location.append(i['user']['location'])
            date.append(i['created_at'])
            tweet.append(i['full_text'])
            try:
                users_mention.append(i['entities']['user_mentions'][0]['screen_name'])
            except IndexError as e:
                users_mention.append('Nan')
            try:
                hashtag.append(i['entities']['hashtags'][0]['text'])
            except IndexError as e:
                hashtag.append('Nan')
            retweet_count.append(i['retweet_count'])
            retweeted.append(i['retweeted'])
            if i['in_reply_to_status_id']:
                comment.append(1)
            else:
                comment.append(0)
            following.append(i['user']['friends_count'])
            dict={'date_created':date,'user':user,'tweet':tweet,'user_mention':users_mention,'retweet_count':retweet_count
                ,'retweeted':retweeted,'location':location,'followers':followers,'following':following,'verfied':verified,'hashtag':hashtag,'comment':comment,'likes':fav_count}
            df=pd.DataFrame(dict)
            try:
                f_A=pd.concat([df,f_A])
            except NameError:
                f_A=df
    
  
    

```


```python
f_A.to_csv('African Leaders.csv')
```


```python
#f_A=pd.read_csv(r'C:\Users\Hp\Downloads\African Leaders.csv')
#I_A=pd.read_csv(r'C:\Users\Hp\Downloads\Influencer.csv')
```


```python


newfile=pd.concat([f_A,I_A])
a=Counter(newfile.hashtag)
del a['Nan']
a={k: v for k, v in sorted(a.items(),reverse=True, key=lambda item: item[1])}

a=a.items()
a=[i[0] for i in a]
record_I={i:'x' for i in a[:5]}
for i in a[:5]:
    record_I[i]=len(I_A.loc[I_A['hashtag']==i])
record_A={i:'x' for i in a[:5]}
for i in a[:5]:
    record_A[i]=len(f_A.loc[f_A['hashtag']==i])
```


```python
record_I=list(record_I.values())
record_A=list(record_A.values())
sum_T=sum(record_I)+sum(record_A)
record_I=[100*i/sum(record_I) for i in record_I]
record_A=[100*i/sum(record_A) for i in record_A]

```


```python
labels=a[:5]
x = np.arange(len(labels))  # the label locations
width = 0.35
plt.style.use('seaborn')
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, record_I, width, label='Government')
rects2 = ax.bar(x + width/2, record_A, width, label='Influencer')
ax.set_ylabel('HashTags')
ax.set_title('Fraction of influencers and gov officials by hashtag')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


#autolabel(rects1)
#autolabel(rects2)

fig.tight_layout()
fig.savefig('final plot.jpg')

plt.show()
```
