import urllib2
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from collections import defaultdict
from string import punctuation
from heapq import nlargest
import nltk


def get_all_doxydonkey_posts(url, links):
    request = urllib2.Request(url)
    response = urllib2.urlopen(request)
    soup = BeautifulSoup(response, 'lxml')
    count = 1;
    for a in soup.find_all('a'):
        try:
            url = a['href']
            title = a['title']
            if title == "Older Posts":
                # print title, url
                links.append(url)
                get_all_doxydonkey_posts(url, links)
                count += 1
        except:
            title = " "
            count > 3
    return

blog_url = "http://doxydonkey.blogspot.in/"
links = []
get_all_doxydonkey_posts(blog_url, links)


def get_doxydonkey_text(test_url):
    request = urllib2.Request(test_url)
    response = urllib2.urlopen(request)
    soup = BeautifulSoup(response, 'lxml')
    my_divs = soup.find_all("div", {"class" : "post-body"})

    posts = []
    for div in my_divs:
        posts += map(lambda p: p.text.encode('ascii', errors='replace').replace("?", " "), div.findAll("li"))
    return posts

doxy_donkey_posts = []
# links.append("http://doxydonkey.blogspot.in/")
for link in links:
    doxy_donkey_posts += get_doxydonkey_text(link)

# print doxy_donkey_posts

vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, stop_words='english')
X = vectorizer.fit_transform(doxy_donkey_posts)

# print X[0]

km = KMeans(n_clusters=3, init='k-means++', max_iter=100, n_init=1, verbose=True)

km.fit(X)

np.unique(km.labels_, return_index=True)

text = {}
for i, cluster in enumerate(km.labels_):
    one_document = doxy_donkey_posts[i]
    if cluster not in text.keys():
        text[cluster] = one_document
    else:
        text[cluster] += one_document

_stopwords = set(stopwords.words('english') + list(punctuation) + ["million", "billion", "year", "millions", "billions", "'s", "''"])

key_words = {}
counts = {}
for cluster in range(3):
    word_sent = word_tokenize(text[cluster].lower())
    word_sent =[word for word in word_sent if word not in _stopwords]
    freq = FreqDist(word_sent)
    key_words[cluster] = nlargest(100, freq, key=freq.get)
    counts[cluster] = freq

unique_keys = {}
for cluster in range(3):
    other_clusters = list(set(range(3))-set([cluster]))
    keys_other_clusters = set(key_words[other_clusters[0]]).union(set(key_words[other_clusters[1]]))
    unique = set(key_words[cluster]) - keys_other_clusters
    unique_keys[cluster] = nlargest(10, unique, key=counts[cluster].get)

# print unique_keys

article = "This weekend, Uber and Lyft — in their reactions to the Trump administration’s immigration order — illustrated how important companies' political views have become to consumers. Lyft took a public stand against the order and, on Sunday, saw more downloads than Uber for the first time ever, according to analysis firm App Annie. Lyft's Sunday downloads also more than doubled its daily average over the previous two weeks. Uber, on the other hand, had a bad weekend. Hundreds of people called for ride-sharers to ditch the company through the hashtag “#deleteUber” after it announced that it would drop surge pricing for John F. Kennedy Airport trips. Many saw Uber’s move as an attempt to undermine the strike that New York City cabdrivers organized to protest the immigration order and capitalize off the controversy — something Uber was quick to deny. It also didn't help Uber's standing among President Trump's critics that its chief executive is on the administration's business advisory committee. The social reaction to the Uber-Lyft divide was immediate. App Annie confirmed that Lyft climbed the app charts on both Apple and Android phones this weekend. It overtook Uber to reach No. 1 on the Apple App Store. That bump came despite the fact that Lyft didn’t suspend its service during the strike either and despite Lyft's ties to another close Trump ally, investor Peter Thiel. Its pledge over the weekend, however, seemed to speak louder than those facts — and louder than a later Uber vow to devote $3 million to help its drivers with immigration legal costs."

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=10)
classifier.fit(X, km.labels_)

test = vectorizer.transform([article.decode('utf-8').encode('ascii', errors='ignore')])
print classifier.predict(test)
