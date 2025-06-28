import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import requests




# Load the movie dataset
data = pd.read_csv('data/movies.csv')

# Fill any missing values in 'overview' column
data['overview'] = data['overview'].fillna('')

# Initialize the TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform the 'overview' column to create a TF-IDF matrix
tfidf_matrix = tfidf_vectorizer.fit_transform(data['overview'])

# Define a function to get movie recommendations
def get_recommendations(title, data, tfidf_matrix, top_n=5):
    if title not in data['title'].values:
        return f"Movie '{title}' not found in the dataset."

    idx = data[data['title'] == title].index[0]
    movie_tfidf_vector = tfidf_matrix[idx]
    cosine_sim = cosine_similarity(movie_tfidf_vector, tfidf_matrix).flatten()
    similar_indices = cosine_sim.argsort()[-top_n-1:-1][::-1]
    movie_details = data[['title', 'genres', 'release_date', 'overview']].drop_duplicates().set_index('title').loc[data['title'].iloc[similar_indices]].reset_index()
    return movie_details

# Define a function to get movie suggestions from TMDb API
def get_external_movie_suggestions(query, api_key, top_n=5):
    url = f'https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={query}&include_adult=false'
    response = requests.get(url)
    data = response.json()
    movies = data.get('results', [])
    
    suggestions = []
    for movie in movies[:top_n]:
        title = movie.get('title', 'No Title')
        overview = movie.get('overview', 'No Overview')
        poster_path = movie.get('poster_path')
        suggestions.append((title, overview, poster_path))
    
    return suggestions


st.set_page_config(page_title="Movie Recommendation", layout="wide", page_icon="🎥")

st.title("🎬 Movie Recommendation System")
st.markdown("**Find movies similar to your favorite one from our database**")

# User input for favorite movie
movie_title = st.text_input("What's your favorite movie:", "")



if st.button('Get Recommendations'):
    if movie_title:
        # Get movie recommendations from the dataset
        recommended_movies = get_recommendations(movie_title, data, tfidf_matrix, top_n=5)
        
        # Display dataset-based recommendations
        if isinstance(recommended_movies, pd.DataFrame) and len(recommended_movies) > 0:
            st.subheader(f"Movies similar to '{movie_title}' (From Dataset):")
            for i, row in recommended_movies.iterrows():
                st.write(f"{i+1}. **{row['title']}**")
                st.write(f"   - Genres: {row['genres']}")
                st.write(f"   - Release Date: {row['release_date']}")
                st.write(f"   - Overview: {row['overview']}")
                st.write("---")
        else:
            st.write(recommended_movies)  # Print the error message if the movie is not found


        


# Adding a footer with extra information
st.markdown("---")
st.markdown("**Developed by [Anchit Das](https://www.linkedin.com/in/itsanchitdas/)**")
st.markdown("**[Github](https://github.com/an-admin)**")
