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

st.image("spencer-davis-ilQmlVIMN4c-unsplash.jpg")
st.caption('Photo by <a href="https://unsplash.com/@spencerdavis?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Spencer Davis</a> on <a href="https://unsplash.com/s/photos/athens?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>', unsafe_allow_html=True)
st.markdown("""---""")
result_container = st.container()

query = st.text_input("Describe your perfect hotel in Athens:", "walking distance to acropolis, clean rooms, pool")
model = SentenceTransformer('all-MiniLM-L6-v2')

@st.cache(persist=True)

def run_search(query, embeddings):

    #model = SentenceTransformer('all-MiniLM-L6-v2')
    
    query_embedding = model.encode(query, convert_to_tensor=True)

    # We use cosine-similarity and torch.topk to find the top 1 score
    cos_scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
    top_result = torch.topk(cos_scores, k=1)

    return top_result


def run(query):

    hotel_df = pd.read_csv("HotelListInAthens__en2019100120191005.csv")
    #hotel_df.fillna(0)

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
    if str(hotel_dict['price_per_night'].values[0]) == "nan":
        price = "Visit provider for current rate."
    else:
        price = str(hotel_dict['price_per_night'].values[0])
    provider = hotel_dict['booking_provider'].values[0]
    deals = hotel_dict['no_of_deals'].values[0]

    with result_container:
    
        st.header('Best hotel match:')

        st.markdown(**hotel_name**)
        st.markdown('Read ' + str(reviews) + ' reviews and more information about this property on <a href="' + hotel_url + '">Trip Advisor</a>')

        #st.text(hotel_url)
        #st.text("Number of Reviews: " + str(reviews))
        st.text("Current Price: " + price)
        st.text("Booking provider: " + provider)
        st.text("Deals Available: " + str(deals))

        st.text("What other guests are saying about this hotel")

        wordcloud = WordCloud(width = 600, height = 600,
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


if st.button('Find a hotel'):
    run(query)
