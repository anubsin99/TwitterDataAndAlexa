from TwitterSearch import *
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
import pandas as pd
from nltk.corpus import stopwords
from collections import defaultdict
import re

tokenizer = RegexpTokenizer(r'\w+')

# create English stop words list
en_stop = get_stop_words('en')

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()

twitterText = []

try:
    tso = TwitterSearchOrder()
    tso.set_keywords(['sustainable', 'green'])
    tso.set_count(5)

    ts = TwitterSearch(
        consumer_key='2k5wzPMTJsuEPuPoKjSk4DpHh',
        consumer_secret='uC1S7lzpoEmaQYoquntqQVzi8FoekqKu2bS5U4JWNWC4G3vv4w',
        access_token='1623061854-86AfxRz021rsfq1dQCIR5JmQo26VNRXHi01UMkE',
        access_token_secret='o3fOUQlTVh5ozone7VjORvm2wG3QrAhjNBdyMHDuRsXzb'
    )

    for tweets in ts.search_tweets_iterable(tso):
        twitterText.append(tweets['text'])
except TwitterSearchException as e:
    print(e)

print(len(twitterText))
print("Got here")

documents = twitterText
documents = [' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z# \t])|(\w+:\/\/\S+)", " ", doc.strip()).split()) for doc in
             documents]
documents = [' '.join([w for w in doc.split() if not w.isdigit() and len(w) > 1]) for doc in documents]
print('Done Reading Data')

# Remove stopwords
cached_stopwords = set(stopwords.words('english'))
cached_stopwords.update(
    [word for line in open('stopwords.txt', 'r') for word in line.split()])  # read from stopwords file
cached_stopwords.update(
    ['rt', 'https', 'http', 'htt', 'gt', 'p', 'amp', 'a', 'about', 'above', 'above', 'across', 'after', 'afterwards',
     'again', 'against', 'all', 'almost', 'alone', 'along', 'already', 'also', 'although', 'always', 'am', 'among',
     'amongst', 'amoungst', 'amount', 'an', 'and', 'another', 'any', 'anyhow', 'anyone', 'anything', 'anyway',
     'anywhere', 'are', 'around', 'as', 'at', 'back', 'be', 'became', 'because', 'become', 'becomes', 'becoming',
     'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides', 'between', 'beyond', 'bill',
     'both', 'bottom', 'but', 'by', 'call', 'can', 'cannot', 'cant', 'co', 'con', 'could', 'couldnt', 'cry', 'de',
     'describe', 'detail', 'do', 'done', 'dont', 'down', 'due', 'during', 'each', 'eg', 'eight', 'either', 'eleven',
     'else', 'elsewhere', 'empty', 'enough', 'etc', 'even', 'ever', 'every', 'everyone', 'everything', 'everywhere',
     'except', 'few', 'fifteen', 'fify', 'fill', 'find', 'fire', 'first', 'five', 'for', 'former', 'formerly', 'forty',
     'found', 'four', 'from', 'front', 'full', 'further', 'get', 'give', 'go', 'had', 'has', 'hasnt', 'have', 'he',
     'hence', 'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his',
     'how', 'however', 'hundred', 'ie', 'if', 'in', 'inc', 'indeed', 'interest', 'into', 'is', 'it', 'its', 'itself',
     'keep', 'last', 'latter', 'latterly', 'least', 'less', 'ltd', 'made', 'many', 'may', 'me', 'meanwhile', 'might',
     'mill', 'mine', 'more', 'moreover', 'most', 'mostly', 'move', 'much', 'must', 'my', 'myself', 'name', 'namely',
     'neither', 'never', 'nevertheless', 'next', 'nine', 'no', 'nobody', 'none', 'noone', 'nor', 'not', 'nothing',
     'now', 'nowhere', 'of', 'off', 'often', 'on', 'once', 'one', 'only', 'onto', 'or', 'other', 'others', 'otherwise',
     'our', 'ours', 'ourselves', 'out', 'over', 'own', 'part', 'per', 'perhaps', 'please', 'put', 'rather', 're',
     'same', 'see', 'seem', 'seemed', 'seeming', 'seems', 'serious', 'several', 'she', 'should', 'show', 'side',
     'since', 'sincere', 'six', 'sixty', 'so', 'some', 'somehow', 'someone', 'something', 'sometime', 'sometimes',
     'somewhere', 'still', 'such', 'system', 'take', 'ten', 'than', 'that', 'the', 'their', 'them', 'themselves',
     'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein', 'thereupon', 'these', 'they', 'thickv',
     'thin', 'third', 'this', 'those', 'though', 'three', 'through', 'throughout', 'thru', 'thus', 'to', 'together',
     'too', 'top', 'toward', 'towards', 'twelve', 'twenty', 'two', 'un', 'under', 'until', 'up', 'upon', 'us', 'very',
     'via', 'was', 'we', 'well', 'were', 'what', 'whatever', 'when', 'whence', 'whenever', 'where', 'whereafter',
     'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while', 'whither', 'who', 'whoever',
     'whole', 'whom', 'whose', 'why', 'will', 'with', 'within', 'without', 'would', 'yet', 'you', 'your', 'yours',
     'yourself', 'yourselves', 'the'])

texts = [[word for word in document.lower().split() if word not in cached_stopwords] for document in documents]
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1
texts = [[token for token in text if frequency[token] > 1] for text in
         texts]  # keep only the words that occured atleast twice in the dataset

# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(texts)

# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts]

# generate LDA model
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=20)
print(ldamodel.print_topics(num_topics=5, num_words=20))

bmlda = ldamodel[corpus]
for t in bmlda:
    print(t)