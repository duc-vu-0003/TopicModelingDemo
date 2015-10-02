import string
import re
import codecs
from os import path
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import wordpunct_tokenize

class Tokenizer:

  def __init__(self):
    self.stemmer = PorterStemmer()
    stopwordsReuter = [line.strip() for line in open(Paths.stopword, 'r')]

    self.stopwords = (stopwordsReuter +
      list(string.punctuation) +
      ['lt', 'gt', 'vs'])

    # self.stopwords = (nltk.corpus.stopwords.words('english') +
    #   list(string.punctuation) +
    #   ['lt', 'gt', 'vs'])
    # self.stopwords = stopwordsReuter
    self.numbers = re.compile('^\d+(st|nd|rd|th|p|pct)?$', re.IGNORECASE)
    self.len_file = 3

  #Tokenize
  #Unicode
  #Remove stopwords, numbers and common abbreviations
  # More things to try:
  # > better tokenising for in-word punctation like u.s. and
  #    energy/usa
  def tokenize(self, doc):
    tokens = wordpunct_tokenize(doc.strip())
    return [word.lower() for word in tokens if word.lower() not in self.stopwords and not self.numbers.match(word) and len(word) >= self.len_file]
    # return [unicode(self.stemmer.stem(t.lower())) for t in tokens
    #         if not t.lower() in self.stopwords and not self.numbers.match(t)]

class SingleFileCorpus(object):

  def __init__(self, in_file, dictionary):
    self.in_file = in_file
    self.dictionary = dictionary

  def __iter__(self):
    with codecs.open(self.in_file, 'r', 'utf-8') as f:
      for doc in f:
        yield self.dictionary.doc2bow(doc.strip().split())

class Paths(object):
  base = 'data'
  reuters = 'reuters'
  texts_clean = path.join(base, 'reuters_preprocessed.txt')
  text_index = path.join(base, 'reuters_fileids.txt')
  dictionary = path.join(base, 'dictionary.txt')
  corpus = path.join(base, 'corpus.mm')
  tfidf_model = path.join(base, 'tfidf.model')
  lsi_model = path.join(base, 'lsi.model')
  similarity_index = path.join(base, 'tfidf-lsi.index')
  stopword = path.join(reuters, 'stopwords')
  lda_model = path.join(base, 'lda.model')
  reuter_test = path.join(base, 'reuters_test.txt')
  final_topics = path.join(base, 'final_topics.csv')
