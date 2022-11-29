#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: David Kloepper (kloe0021@umn.edu)
"""

import pandas as pd
import matplotlib.pyplot as plt

import pickle as pkl
from sentence_transformers import SentenceTransformer, util

import torch

from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)

import streamlit as st

st.title("Athens Hotel Search")

st.image("spencer-davis-ilQmlVIMN4c-unsplash.jpg", caption='Photo by <a href="https://unsplash.com/@spencerdavis?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Spencer Davis</a> on <a href="https://unsplash.com/s/photos/athens?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>')

#st.markdown("This is a demo Streamlit app on the web.")
#st.markdown("My name is David, hello world!..")
#st.markdown("This is v0.1")

query = st.text_input("Describe your perfect hotel in Athens:", "near akropolis")
model = SentenceTransformer('all-MiniLM-L6-v2')

@st.cache(persist=True)

def run_search(query, embeddings):

    #model = SentenceTransformer('sentence-transformers/paraphrase-xlm-r-multilingual-v1')
    #model = SentenceTransformer('all-MiniLM-L6-v2')
    
    query_embedding = model.encode(query, convert_to_tensor=True)

    # We use cosine-similarity and torch.topk to find the top 1 score
    cos_scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
    top_result = torch.topk(cos_scores, k=1)

    #results = list(zip(top_results[0], top_results[1]))

    #return results

    return top_result


def run(query):

    #hotel_df = pd.read_csv("https://raw.githubusercontent.com/dkloepper/MABA6490-Assignment2/3c4443422597a40d0b9cc7115ca8d5edc11d609f/HotelListInAthens__en2019100120191005.csv")
    hotel_df = pd.read_csv("HotelListInAthens__en2019100120191005.csv")
    hotel_df.fillna(value=0)

    with open('athens-embeddings.pkl', 'rb') as fIn:
        corpus_embedding = pkl.load(fIn)

    embeddings = corpus_embedding['embeddings']
    corpus = corpus_embedding['sentences']
    reviews_df = corpus_embedding['reviews']

    search_result = run_search(query, embeddings)

    score = search_result[0]
    idx = search_result[1]

    review_dict = reviews_df.loc[reviews_df['review_concat'] == corpus[idx]]

    reco_hotel = review_dict['hotelName'].values[0]
    hotel_dict = hotel_df.loc[hotel_df['hotel_name']== reco_hotel]

    hotel_name = hotel_dict['hotel_name'].values[0]
    hotel_url = hotel_dict['url'].values[0]
    reviews = hotel_dict['reviews'].values[0]
    #price = hotel_dict['price_per_night'].values[0]
    if hotel_dict['price_per_night'].values[0] == 0:
        price = "Visit provider for current rate."
    else:
        price = str(hotel_dict['price_per_night'].values[0])
    provider = hotel_dict['booking_provider'].values[0]
    deals = hotel_dict['no_of_deals'].values[0]

    st.text(hotel_name)
    st.text(hotel_url)
    st.text("Number of Reviews: " + str(reviews))
    st.text("Current Price: " + price)
    st.text("Booking provider: " + provider)
    st.text("Deals Available: " + str(deals))

    wordcloud = WordCloud(width = 800, height = 800,
        background_color ='white',
        stopwords = stopwords,
        min_font_size = 10).generate(corpus[idx])
 
    # plot the WordCloud image                      
    fig = plt.figure(figsize = (8, 8), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    
    #plt.show()

    st.pyplot(fig)


if st.button('Start search'):
    run(query)
