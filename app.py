#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: David Kloepper (kloe0021@umn.edu)
"""

import pandas as pd
import matplotlib.pyplot as plt

import pickle as pkl
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')

import torch

from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)

import streamlit as st

body_container = st.container()
result_container = st.container()

with body_container:
    st.title("Athens Hotel Search")
    st.image("spencer-davis-ilQmlVIMN4c-unsplash.jpg")
    st.caption('Photo by <a href="https://unsplash.com/@spencerdavis?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Spencer Davis</a> on <a href="https://unsplash.com/s/photos/athens?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>', unsafe_allow_html=True)
    st.markdown("""---""")
    query = st.text_input("Describe your perfect hotel in Athens:", "walking distance to acropolis, clean rooms, pool")
    search_button = st.button('Find a hotel')

@st.cache(persist=True)

def run_search(query, embeddings):
    
    query_embedding = model.encode(query, convert_to_tensor=True)

    # We use cosine-similarity and torch.topk to find the top 1 score
    cos_scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
    top_result = torch.topk(cos_scores, k=1)

    return top_result


def run(query):

    hotel_df = pd.read_csv("HotelListInAthens__en2019100120191005.csv")

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
    provider = hotel_dict['booking_provider'].values[0]
    if str(hotel_dict['price_per_night'].values[0]) == "nan":
        #price = "Visit provider for current rate."
        price = 'Visit <a href="' + provider + '">' + provider + '</a> for current rate.'
    else:
        #price = str(hotel_dict['price_per_night'].values[0])
        price = str(hotel_dict['price_per_night'].values[0]) + 'from <a href="' + provider + '">' + provider + '</a>.'
    
    deals = hotel_dict['no_of_deals'].values[0]

    with result_container:
    
        st.header('Best hotel match:')

        st.subheader(hotel_name)
        st.markdown("Best available price: " + price, unsafe_allow_html=True)
        st.markdown("See " + deals + ' additional deals from <a href="' + provider + '">' + provider + '</a>.',unsafe_allow_html=True)
        st.text("")
        st.markdown('Read ' + str(reviews) + ' reviews and more information about this property on <a href="' + hotel_url + '">Trip Advisor</a>', unsafe_allow_html=True)

        #st.markdown("Current Price: " + price)
        #st.text("Booking provider: " + provider)
        #st.text("Deals Available: " + str(deals))

        st.markdown("What other guests are saying about this hotel:")

        wordcloud = WordCloud(width = 300, height = 300,
            background_color ='white',
            stopwords = stopwords,
            min_font_size = 10).generate(corpus[idx])
    
        # plot the WordCloud image                      
        fig = plt.figure(figsize = (8, 8), facecolor = None)
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.tight_layout(pad = 0)
        
        st.pyplot(fig)


if search_button:
    run(query)
