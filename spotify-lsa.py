# imports 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import spotipy
import spotipy.util as util
import sys
from datetime import date
import lyricsgenius as genius
from gensim import corpora
from gensim.models import LsiModel
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim.models.coherencemodel import CoherenceModel
import re 
import wordcloud


# LSA docs: https://www.datacamp.com/community/tutorials/discovering-hidden-topics-python

# get data from spotify API 

# get data from lyricsgenius API 

#username = input("Enter spotify email: ")
scope = 'user-top-read'
token = util.prompt_for_user_token(username,scope,client_id,client_secret,redirect_uri)

# create objects 
sp = spotipy.Spotify(auth=token)
api = genius.Genius(lg_key)
sp.trace = False

name = input("Enter Artist Name: ")


# LSA 
def get_song_names_from_artist(name):  
    result = sp.search(name)
    artist_info = result['tracks']['items'][0]['artists']
    a = artist_info[0]
    uri = a['uri']
    
    results = sp.artist_top_tracks(uri)
    d = results.get('tracks')
    song_names = []
    for i in  range(len(d)): 
        s = d[i]
        n = s['name']
        song_names.append(n)
    return song_names

artist_songs = get_song_names_from_artist(name)

lyrics = {}
# get the lyrics for these song names
for song in artist_songs: 
    get_song = api.search_song(song, name)
    l = get_song.lyrics 
    l=l.strip().replace('\n', ' ')
    l = re.sub(r'\[(.*?)\]', " ", l) # remove verses, intro
    lyrics[song] = l.rstrip() 



combined_lyrics = []
for song in lyrics.keys(): 
    c = lyrics[song]
    combined_lyrics.append(c)


def preprocess_data(doc_set):
    """
    Input  : docuemnt list
    Purpose: preprocess text (tokenize, removing stopwords, and stemming)
    Output : preprocessed text
    """
    # initialize regex tokenizer
    tokenizer = RegexpTokenizer(r'\w+')
    # create English stop words list
    en_stop = set(stopwords.words('english'))
    # Create p_stemmer of class PorterStemmer
    p_stemmer = PorterStemmer()
    # list for tokenized documents in loop
    texts = []
    # loop through document list
    for i in doc_set:
        # clean and tokenize document string
        raw = i.lower()
        tokens = tokenizer.tokenize(raw)
        # remove stop words from tokens
        stopped_tokens = [i for i in tokens if not i in en_stop]
        # stem tokens
        stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
        # add tokens to list
        texts.append(stemmed_tokens)
    return texts


def compute_coherence_values(dictionary, doc_term_matrix, doc_clean, stop, start=2, step=3):
    """
    Input   : dictionary : Gensim dictionary
              corpus : Gensim corpus
              texts : List of input texts
              stop : Max num of topics
    purpose : Compute c_v coherence for various number of topics
    Output  : model_list : List of LSA topic models
              coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, stop, step):
        # generate LSA model
        model = LsiModel(doc_term_matrix, num_topics=number_of_topics, id2word = dictionary)  # train model
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=doc_clean, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values


def prepare_corpus(doc_clean):
     """
     Input  : clean document
     Purpose: create term dictionary of our courpus and Converting list of documents (corpus) into Document Term Matrix
     Output : term dictionary and Document Term Matrix
     """
     # Creating the term dictionary of our courpus, where every unique term is assigned an index. dictionary = corpora.Dictionary(doc_clean)
     dictionary = corpora.Dictionary(doc_clean)
     # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
     doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
     # generate LDA model
     return dictionary,doc_term_matrix   

def create_gensim_lsa_model(doc_clean,number_of_topics,words):
    """
    Input  : clean document, number of topics and number of words associated with each topic
    Purpose: create LSA model using gensim
    Output : return LSA model
    """
    dictionary,doc_term_matrix=prepare_corpus(doc_clean)
    # generate LSA model
    lsamodel = LsiModel(doc_term_matrix, num_topics=number_of_topics, id2word = dictionary)  # train model
    topics = lsamodel.print_topics(num_topics=number_of_topics, num_words=words)
    return lsamodel, topics

# evaluate model 
def plot_graph(doc_clean,start, stop, step):
    dictionary,doc_term_matrix=prepare_corpus(doc_clean)
    model_list, coherence_values = compute_coherence_values(dictionary, doc_term_matrix,doc_clean,
                                                            stop, start, step)
    # Show graph
    x = range(start, stop, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()



# start with one song 
number_of_topics = 4
words=10
clean_text=preprocess_data(combined_lyrics)
model, topics = create_gensim_lsa_model(clean_text,number_of_topics,words)
start,stop,step=2,10,1
plot_graph(clean_text,start,stop,step)
print(topics)
