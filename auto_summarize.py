import urllib2

from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from heapq import nlargest
from nltk.probability import FreqDist
from collections import defaultdict

sample_url = "https://www.washingtonpost.com/national/health-science/trump-gives-no-sign-of-backing-down-from-travel-ban/2017/01/29/4ffe900a-e620-11e6-b82f-687d6e6a3e7c_story.html?hpid=hp_hp-top-table-main_banledeall-917am%3Ahomepage%2Fstory&utm_term=.211af7fd6e4e"


def get_text_from_url(article_url):
    page = urllib2.urlopen(article_url).read().decode('utf8', 'ignore')
    soup = BeautifulSoup(page, 'lxml')
    # print(soup.find('article').text)
    joined_text = ' '.join(map(lambda p: p.text, soup.find_all('article')))
    return joined_text.encode('ascii', errors='replace').replace("?", " ")

text = get_text_from_url(sample_url)
sents = sent_tokenize(text)

word_sent = word_tokenize(text.lower())

_stopwords = set(stopwords.words('english')+list(punctuation))

word_sent=[word for word in word_sent if word not in _stopwords]

freq = FreqDist(word_sent)
# print nlargest(10, freq, key=freq.get)

ranking = defaultdict(int)

for i, sent in enumerate(sents):
    for w in word_tokenize(sent.lower()):
        if w in freq:
            ranking[i] += freq[w]

sents_index = nlargest(4, ranking, ranking.get)
# print sents_index

print [sents[j] for j in sorted(sents_index)]