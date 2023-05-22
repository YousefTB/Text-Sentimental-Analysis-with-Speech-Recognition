#!/usr/bin/env python
# coding: utf-8

# In[97]:


import numpy as np
import pandas as pd
import nltk
from gensim.parsing.preprocessing import remove_stopwords, STOPWORDS
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import speech_recognition as sr
import warnings
warnings.filterwarnings("ignore")


# In[67]:


# Defining all Required Paramaters for WordNet Lemmatizer (POS Tags Reducing - POS Tagging)
lemmatizer = WordNetLemmatizer()
# pos_tagger function to Reduce Nouns to N, Adjectives to J ... etc , for further processing
def pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:         
        return None
    
def lemmatization(text):
    new_sen = []
    pos_tagged = pos_tag(text)
    wordnet_tagged = list(map(lambda x: (x[0],pos_tagger(x[1])), pos_tagged)) # POS Tagging & Reducing
    for word, tag in wordnet_tagged:
        if tag is None:
            new_sen.append(word)
        else:
            new_sen.append(lemmatizer.lemmatize(word,tag))                   # Lemmatizing
            
    return new_sen


# In[98]:


LGR_Model = pickle.load(open('LGR_Model.sav','rb'))


# In[96]:


r = sr.Recognizer()
with sr.Microphone() as source:
    print("Say Something !")
    r.adjust_for_ambient_noise(source)
    audio = r.listen(source)
    
Sentence = r.recognize_google(audio)   # The sentence to be tested

# Firstly Preprocess the sentence
lower_sen = Sentence.lower()                                  # Lowering all characters
cleaned_sentence = remove_stopwords(lower_sen)                # Removing Stop Words
tken_sentence = word_tokenize(cleaned_sentence)               # Tokenizing to be Lemmatized
lemmatized_sen = lemmatization(tken_sentence)                 # Lemmatization with POS
processed_sen = ' '.join(lemmatized_sen)                      # Regrouping

# Display the Sentence
print(processed_sen)

# Lets Predict the sentence if POSITIVE or NEGATIVE
sentiment = LGR_Model.predict([processed_sen])                       # Prediction
if sentiment[0] == 1:
    print(Sentence, '(is Positive)')
else:
    print(Sentence, '(is Negative)')

