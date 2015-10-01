import codecs
from nltk.corpus import reuters
from utils import Tokenizer, Paths, SingleFileCorpus
from gensim import corpora, models, similarities
from nltk.tokenize import word_tokenize, wordpunct_tokenize
from nltk.stem.porter import PorterStemmer
import numpy as np
import os
import logging
from nltk.stem.wordnet import WordNetLemmatizer
import random
import nltk

# Preprocess script - build a single text file with cleaned, normalised documents
#  - tokenised, stemmed, one document per line.
# Track fileids to retrieve document text later
def preProcess():
    docs = 0
    bad = 0
    tokenizer = Tokenizer()

    if not os.path.isdir(Paths.base):
        os.makedirs(Paths.base)

    with open(Paths.text_index, 'w') as fileid_out:
      with codecs.open(Paths.texts_clean, 'w', 'utf-8-sig') as out:
          with codecs.open(Paths.reuter_test, 'w', 'utf-8-sig') as test:
              for f in reuters.fileids():
                  contents = reuters.open(f).read()
                  try:
                    tokens = tokenizer.tokenize(contents)
                    # tokens = contents

                    docs += 1
                    if docs % 1000 == 0:
                      print "Normalised %d documents" % (docs)
                      out.write(' '.join(tokens) + "\n")
                    # if f.startswith("train"):
                    #
                    # else:
                    #     test.write(' '.join(tokens) + "\n")
                    fileid_out.write(f + "\n")

                  except UnicodeDecodeError:
                    bad += 1

    print "Normalised %d documents" % (docs)
    print "Skipped %d bad documents" % (bad)
    print 'Finished building train file ' + Paths.texts_clean
    # print 'Finished building test file ' + Paths.reuter_test

def buildCorpus():
    # Build corpus script - build gensim dictionary and corpus
    dictionary = corpora.Dictionary()

    print "Create dictionary and write out a processed file with one document per line"
    #First pass: create dictionary and write out a processed file with
    # one document per line
    with codecs.open(Paths.texts_clean, 'r', 'utf-8') as f:
      for doc in f:
        tokens = doc.strip().split()
        dictionary.doc2bow(tokens, allow_update=True)

    print "Remove very rare and very common words"
    # Remove very rare and very common words
    dictionary.filter_extremes(no_below=1, no_above=0.8)
    dictionary.save(Paths.dictionary)

    print "Second pass over files to serialize corpus to file"
    #Second pass over files to serialize corpus to file
    corpus = SingleFileCorpus(Paths.texts_clean, dictionary)
    corpora.MmCorpus.serialize(Paths.corpus, corpus)

def trainLDA(n_topics):
    print "Loading corpus and dictionary"
    corpus = corpora.MmCorpus(Paths.corpus)
    dictionary = corpora.Dictionary.load(Paths.dictionary)

    print "extract %d LDA topics, using 20 full passes, no online updates" % (n_topics)
    lda = models.LdaModel(corpus, id2word=dictionary, num_topics=n_topics, update_every=0, passes=20)
    print "Saving LDA Model"
    lda.save(Paths.lda_model)
    print 'Finished train LDA Model %s ' + Paths.lda_model

def displayLDA(n_topics, num_words):
    print "Loading LDA Model"
    lda = models.LdaModel.load(Paths.lda_model)
    i = 0
    # show_topics(num_topics=10, num_words=10, log=False, formatted=True)
    # for topic in lda.show_topics(num_topics=n_topics, num_words=num_words, log=False, formatted=True):
    #     print '#' + str(i) + ': ' + topic
    #     i += 1

    topics_matrix = lda.show_topics(formatted=False, num_words=num_words)
    topics_matrix = np.array(topics_matrix)

    topic_words = topics_matrix[:,:,1]
    for topic in topic_words:
        i += 1
        print 'Topic: ', i
        print([str(word) for word in topic])

def load_stopwords():
    print "Loading Stop Words List"
    stopwords = {}
    with open(Paths.stopword, 'rU') as f:
        for line in f:
            stopwords[line.strip()] = 1
    return stopwords

def extract_lemmatized_nouns(new_review):
    print "Start tagging to get Noun words"
    stopwords = load_stopwords()
    words = []

    sentences = nltk.sent_tokenize(new_review.lower())
    for sentence in sentences:
        tokens = nltk.word_tokenize(sentence)
        text = [word for word in tokens if word not in stopwords]
        tagged_text = nltk.pos_tag(text)

        for word, tag in tagged_text:
            words.append({"word": word, "pos": tag})

    lem = WordNetLemmatizer()
    nouns = []
    for word in words:
        if word["pos"] in ["NN", "NNS"]:
            nouns.append(lem.lemmatize(word["word"]))

    print "Finish POS"
    return nouns

def predict(new_topic):
    print "Loading model and dictionary"
    dictionary = corpora.Dictionary.load(Paths.dictionary)
    lda = models.LdaModel.load(Paths.lda_model)
    # transform into LDA space
    print "Preprocessing new data"
    tokens = extract_lemmatized_nouns(new_topic)
    # tokens = new_topic.strip().split()
    new_topic_bow = dictionary.doc2bow(tokens)
    new_topic_lda = lda[new_topic_bow]

    print(new_topic_lda)
    # print the document's single most prominent LDA topic
    print(lda.print_topic(max(new_topic_lda, key=lambda item: item[1])[0]))

def loadTestTopic(number_topic):
    with open(Paths.reuter_test, 'r') as f:
        data = f.read().split('\n')
    random.shuffle(data)
    return data[:number_topic]

def main():
    oper = -1
    while int(oper) != 0:
        print('**************************************')
        print('Choose one of the following: ')
        print('1 - PreProcess Data')
        print('2 - Build Corpus')
        print('3 - Train LDA')
        print('4 - Display LDA Topic')
        print('5 - Predict Topics')
        print('0 - Exit')
        print('**************************************')
        oper = int(input("Enter your options: "))

        if oper == 0:
            exit()
        elif oper == 1:
            preProcess()
        elif oper == 2:
            buildCorpus()
        elif oper == 3:
            trainLDA(100)
        elif oper == 4:
            displayLDA(10, 100)
        elif oper == 5:
            test_data = loadTestTopic(2)
            for new_topic in test_data:
                print new_topic
                print "-----------------"
                predict(new_topic)
                print "\n"

if __name__ == "__main__":
    main()
