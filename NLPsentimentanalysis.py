#sentiment analysis using Natural Language Toolkit

import nltk
#nltk.download()
from nltk import word_tokenize, pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

lem = WordNetLemmatizer()
stem = PorterStemmer()

print(lem.lemmatize("dancing","v"))
print(stem.stem("dancing"))

phrase = "Today I started learning NLP!"
tokens = word_tokenize(phrase)
tokens2 = phrase.split(" ")

print(pos_tag(tokens))

from nltk.corpus import gutenberg
from nltk import ngrams
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def removeNoise(input_text):
    noise_list = ["is", "a","this","...", "we", "When"]
    words = input_text.split()
    noise_free_words = [word for word in words if word not in noise_list]
    noise_free_text = " ".join(noise_free_words)
    return noise_free_text

nltk.corpus.gutenberg.fileids()
macbeth_sentences = gutenberg.sents('shakespeare-macbeth.txt')
macbeth = gutenberg.raw('shakespeare-macbeth.txt')
macbeth = removeNoise(macbeth[0:200])
#print(macbeth)

sid = SentimentIntensityAnalyzer()
ss = sid.polarity_scores("this sucks!")
#print(ss)

grams = ngrams(macbeth_sentences[9], 3)
print(grams)
for g in grams:
    print(g)
 from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
from nltk.corpus import gutenberg

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

def removeNoise(input_text):
    noise_list = ["is", "a","this","...", "we", "When"]
    words = input_text.split()
    noise_free_words = [word for word in words if word not in noise_list]
    noise_free_text = " ".join(noise_free_words)
    return noise_free_text


macbeth = gutenberg.raw('shakespeare-macbeth.txt')
macbeth = macbeth.split("\n\n")

macbeth = [removeNoise(mac) for mac in macbeth]
#print(macbeth[0:10])
#sid = SIA()
#print(sid.polarity_scores("I want to murder"))

vectorizer = TfidfVectorizer(ngram_range=(1,3))
X = vectorizer.fit_transform(macbeth)
#print(X)

feature_names = vectorizer.get_feature_names()
#print(type(feature_names))

doc=21
feature_index = X[doc,:].nonzero()[1]
tfidf_scores = zip(feature_index, [X[doc,x] for x in feature_index])
for w, s in [(feature_names[i], s) for (i, s) in tfidf_scores]:
    print(w,s)   
  
