import nltk

from nltk.stem.lancaster import LancasterStemmer    #using lancaster stemmer algorithm
from nltk.tokenize import word_tokenize
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn

text2 = "Mary closed on closing night when she was in the mood to close."

st = LancasterStemmer()
stemmedWords = [st.stem(word) for word in word_tokenize(text2)]

print stemmedWords

print(nltk.pos_tag(word_tokenize(text2)))

# for ss in wn.synsets('bass'):
#    print(ss, ss.definition())

sensel = lesk(word_tokenize("sing in a lower tone, along with the bass"), "bass")
print(sensel, sensel.definition())