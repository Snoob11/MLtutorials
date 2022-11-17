#sentiment analysis using nltk

from nltk.corpus import gutenberg
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np

sid = SentimentIntensityAnalyzer()

def sentAnalyze(corpus_name):
    sentDict ={'neg':[],'neu':[],'pos':[],'compound':[]}
    corpus = gutenberg.raw(corpus_name)
    corpus = corpus.split("\n\n")
    
    for corp in corpus:
        corp=corp.strip()
        if corp =="":
            continue
        
        ss = sid.polarity_scores(corp)
        for s in ss:
            sentDict[s].append(ss[s])
    
    for sent in sentDict:
        sentDict[sent]= np.mean(sentDict[sent])
    return sentDict

sid = SentimentIntensityAnalyzer()

ceaDict = sentAnalyze("shakespeare-caesar.txt")
macDict = sentAnalyze("shakespeare-macbeth.txt")

print("Sentiment, Ceasar, Macbeth, Difference")
for sent in ceaDict:
    print(sent, ceaDict[sent], macDict[sent], ceaDict[sent]-macDict[sent])
