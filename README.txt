Learning basic natural language processing and topic modelling techniques with NLTK and Gensim.

Link: https://github.com/sirty/TopicModelingDemo

 - Document tokenising and normalising
 - Removing stopwords and words that are very rare and very common in a corpus
 - Bag-of-words vectors
 - Latent Dirichlet Allocation
 - Learn more about the hidden topics of reuter courpus
 - Display as word cloud
 
Install

-- For basic task
>   $ pip install nltk
>   $ pip install gensim

-- Data for nltk
>   $ python -m nltk.downloader reuters
>   $ python -m nltk.downloader punkt

-- For run WordCloud function
>   $ pip install wordcloud
>   $ pip install Pillow
>   $ pip install pandas

Quick Start
To run this tool
$ python main.py

**************************************
Choose one of the following: 
1 - PreProcess Data
2 - Build Corpus
3 - Train LDA
4 - Display LDA Topic
5 - Show as Word Cloud
0 - Exit
**************************************
Follow options from 1-->5

Step by step:
1. Run the actual text processing (tokenization, removal of stop words)
Input: Reuters documents
Output: List tokenized documents 
    return [word.lower() for word in tokens if word.lower() not in self.stopwords and not self.numbers.match(word) and len(word)     >= self.len_file]
    
2. Do some Gensim specific conversions, also filter out extreme words

#create a Gensim dictionary from the texts
dictionary = corpora.Dictionary(texts)

#remove extremes (similar to the min/max df step used when creating the tf-idf matrix)
dictionary.filter_extremes(no_below=1, no_above=0.8)

#convert the dictionary to a bag of words corpus for reference
corpus = [dictionary.doc2bow(text) for text in texts]

3. Train LDA model
From corpus and dictionary in Step 2 --> train LDA model
In this case , num_topics = 10
lda = models.LdaModel(corpus, id2word=dictionary, num_topics=n_topics, update_every=0, passes=20)

4. Show topics
Each topic has a set of words that defines it, along with a certain probability.


Convert the topics into just a list of the top 100 words in each topic.


