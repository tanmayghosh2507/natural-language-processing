import nltk

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.collocations import *

text = 'Mary had a little lamb. Her fleece was white as snow.'

sents = sent_tokenize(text)
print(sents)

words = [word_tokenize(sent) for sent in sents]
print[words]

customStopWords = set(stopwords.words('english')+list(punctuation))   # import a set of stopwords and punctuation in english language
wordsWOStopwords = [word for word in word_tokenize(text) if word not in customStopWords]
print wordsWOStopwords

bigrams_measures = nltk.collocations.BigramAssocMeasures()
finder = BigramCollocationFinder.from_words(wordsWOStopwords)
print(sorted(finder.ngram_fd.items()))