#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: David Kloepper (kloe0021@umn.edu)
"""
import re
import pandas as pd
import os
import matplotlib.pyplot as plt

import pickle as pkl
from sentence_transformers import SentenceTransformer, util

import torch
import scipy.spatial

from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)

import streamlit as st

st.title("Welcome to MABA Class, David Kloepper")
st.markdown("This is a demo Streamlit app on the web.")
st.markdown("My name is David, hello world!..")
st.markdown("This is v2.1")

@st.cache(persist=True)

def run_search(query, embeddings):

    model = SentenceTransformer('sentence-transformers/paraphrase-xlm-r-multilingual-v1')
    
    query_embedding = model.encode(query, convert_to_tensor=True)

    # We use cosine-similarity and torch.topk to find the top 1 score
    cos_scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
    top_results = torch.topk(cos_scores, k=1)

    results = zip(top_results[0], top_results[1])

    return results


def run():

    hotel_df = pd.read_csv("https://raw.githubusercontent.com/dkloepper/MABA6490-Assignment2/3c4443422597a40d0b9cc7115ca8d5edc11d609f/HotelListInAthens__en2019100120191005.csv")

    with open('https://github.com/dkloepper/MABA6490-Assignment2/blob/2dc97d3b70176d85894a6bf52b80cae4a9ff3233/athens-embeddings.pkl', 'rb') as fIn:
        corpus_embedding = pkl.load(fIn)

    embeddings = corpus_embedding['embeddings']
    corpus = corpus_embedding['sentences']
    reviews_df = corpus_embedding['reviews']

    query = "close to the akropolis"

    search_result = run_search(query, embeddings)

    score = search_result[0]
    idx = search_result[1]

    review_dict = reviews_df.loc[reviews_df['review_concat'] == corpus[idx]]

    reco_hotel = review_dict['hotelName'].values[0]
    hotel_dict = hotel_df.loc[hotel_df['hotel_name']== reco_hotel]

    hotel_name = hotel_dict['hotel_name'].values[0]
    hotel_url = hotel_dict['url'].values[0]
    reviews = hotel_dict['reviews'].values[0]
    price = hotel_dict['price_per_night'].values[0]
    provider = hotel_dict['booking_provider'].values[0]
    deals = hotel_dict['no_of_deals'].values[0]

    print(hotel_name)
    print(score)

    wordcloud = WordCloud(width = 800, height = 800,
        background_color ='white',
        stopwords = stopwords,
        min_font_size = 10).generate(corpus[idx])
 
    # plot the WordCloud image                      
    plt.figure(figsize = (8, 8), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    
    plt.show()

if __name__ == '__main__':
    run()