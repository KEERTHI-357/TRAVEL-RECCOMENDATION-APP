import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Set page configuration for a wider layout
st.set_page_config(layout="wide")

st.title("Travel Recommendation System")
st.markdown("### Find your next favorite destination based on user ratings!")
# --- Data Loading and Preprocessing ---

@st.cache_data
def load_data():
    """Loads and preprocesses all datasets."""
    try:
        cities_df = pd.read_excel('Copy of City.xlsx')
        continents_df = pd.read_excel('Copy of Continent.xlsx')
        countries_df = pd.read_excel('Copy of Country.xlsx')
        items_df = pd.read_excel('Copy of Item.xlsx')
        visiting_mode_df = pd.read_excel('Copy of Mode.xlsx')
        regions_df = pd.read_excel('Copy of Region.xlsx')
        transactions_df = pd.read_excel('Copy of Transaction.xlsx')
        types_df = pd.read_excel('Copy of Type.xlsx')
        users_df = pd.read_excel('Copy of User.xlsx') 
    except FileNotFoundError as e:
        st.error(f"Error: One or more data files not found. Please ensure all CSV files are in the same directory.")
        st.stop()       
    # Merge the dataframes on AttractionId to link user ratings with attraction names
    merged_df = pd.merge(transactions_df, items_df, left_on='AttractionId', right_on='AttractionId', how='inner')
    merged_df.rename(columns={'Attraction_x': 'Attraction'}, inplace=True)
    merged_df.dropna(subset=['Attraction', 'Rating', 'UserId'], inplace=True)

    return merged_df

merged_data = load_data()

# --- Recommendation System Logic ---

@st.cache_resource
def prepare_recommendation_model(df):
    """
    Creates the user-item matrix and calculates user similarity.
    This is the core of the collaborative filtering model.
    """
    # Create a pivot table with users as rows and attractions as columns
    user_item_matrix = df.pivot_table(
        index='UserId',
        columns='Attraction',
        values='Rating'
    ).fillna(0)

    # Calculate the cosine similarity between users
    user_similarity = cosine_similarity(user_item_matrix)
    user_similarity_df = pd.DataFrame(
        user_similarity,
        index=user_item_matrix.index,
        columns=user_item_matrix.index
    )

    return user_item_matrix, user_similarity_df

user_item_matrix, user_similarity_df = prepare_recommendation_model(merged_data)

def get_recommendations(user_id, user_item_matrix, user_similarity_df, top_n=5):
    """
    Generates personalized recommendations for a given user ID.
    If the user is new, it returns popular attractions.
    """
    if user_id not in user_similarity_df.index:
        # For a new or non-existent user, recommend the most popular items
        popular_attractions = merged_data['Attraction'].value_counts().head(top_n).index.tolist()
        return popular_attractions, "User ID not found. Displaying general recommendations based on popularity."

    # Find users similar to the target user
    similar_users = user_similarity_df[user_id].sort_values(ascending=False).index[1:]

    recommendations = {}
    user_rated_items = user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index

    for similar_user in similar_users:
        if user_similarity_df.loc[user_id, similar_user] > 0.1:  # Use a similarity threshold
            similar_user_rated_items = user_item_matrix.loc[similar_user][user_item_matrix.loc[similar_user] > 0]
            unrated_items = similar_user_rated_items[~similar_user_rated_items.index.isin(user_rated_items)]

            for item, rating in unrated_items.items():
                if item not in recommendations:
                    recommendations[item] = 0.0
                # Add the weighted rating to the recommendation score
                recommendations[item] += rating * user_similarity_df.loc[user_id, similar_user]

    # Sort recommendations by score and return the top N
    recommended_items = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [item[0] for item in recommended_items], ""

# --- Streamlit UI ---
st.sidebar.header("Find Recommendations")
user_id_input = st.sidebar.number_input("Enter a User ID (e.g., 14, 16, 20):", min_value=1, value=14, step=1)

if st.sidebar.button("Get Recommendations"):
    st.subheader(f"Top Attractions for User {user_id_input}")
    
    recommendations, message = get_recommendations(user_id_input, user_item_matrix, user_similarity_df, top_n=10)
    
    if message:
        st.info(message)
    
    if recommendations:
        for i, attraction in enumerate(recommendations):
            st.write(f"**{i+1}.** {attraction}")
    else:
        st.warning("No recommendations could be generated for this user. They may have already rated all items, or there may not be similar users.")
