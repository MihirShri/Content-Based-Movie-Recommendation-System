"""
Author @ Mihir_Srivastava
Dated - 09-07-2020
File - Content_Based_Movie_Recommender_System
Aim - To recommend top 10 movies either based on the plot or taking into account the actors, director, genre and
keywords of the last movie watched by the user depending upon the user's choice.
"""

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from ast import literal_eval

# Use the following code snippet to see data of all columns in pycharm (it actually displays '...' after 2 columns if
# there are more than 5 columns to display)
desired_width = 1000
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 10)

# Read csv files
metadata = pd.read_csv('movies_metadata.csv', low_memory=False)
credits = pd.read_csv('credits.csv')
keywords = pd.read_csv('keywords.csv')

# Drop rows with bad IDs (IDs that don't follow the standard criteria)
metadata = metadata.drop([19730, 29503, 35587])

# Create a copy the metadata DataFrame to recommend movies based on plot. The original DataFrame will be used to
# recommend movies based on the actors, director, genre and keywords
df = metadata.copy()

# Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a', 'an', 'and' etc.
tfidf = TfidfVectorizer(stop_words='english')

# Replace NaN with an empty string
df['overview'] = df['overview'].fillna('')

# Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(df['overview'])

# Map the movie titles with their indices
indices1 = pd.Series(df.index, index=df['title']).drop_duplicates()

##################################
# DONE WITH PLOT BASED RECOMMENDER
##################################

# Convert the IDs to int. Required later for merging these DataFrames into one.
keywords['id'] = keywords['id'].astype(int)
credits['id'] = credits['id'].astype(int)
metadata['id'] = metadata['id'].astype(int)

# Merge the DataFrames into a single metadata DataFrame
metadata = metadata.merge(credits, on='id')
metadata = metadata.merge(keywords, on='id')

# Our data is present in the form of "stringified" lists. We need to convert it into a way that is usable for us.
features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    metadata[feature] = metadata[feature].apply(literal_eval)


# A function to get the name of the director (argument x is a column in our DataFrame)
def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    # If the director is not listed, return NaN
    return np.nan


# A function to get the top 3 elements of any list or the entire list, whichever is less (argument x will be the list
# of cast, keywords and genres)
def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        # Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
        if len(names) > 5:
            names = names[:5]
        return names

    # Return empty list in case of missing/malformed data
    return []


# Define new director, cast, genres and keywords features that are in a suitable form
metadata['director'] = metadata['crew'].apply(get_director)
features = ['cast', 'keywords', 'genres']
for feature in features:
    metadata[feature] = metadata[feature].apply(get_list)


# A function to convert all strings to lower case and strip the names of spaces (Because we don't want Johnny Depp and
# Johnny Sins to be the same)
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        # Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''


# Apply clean_data function to required features features
features = ['cast', 'keywords', 'director', 'genres']
for feature in features:
    metadata[feature] = metadata[feature].apply(clean_data)


# This function will simply join all the required columns by a space so that they can be fed into the word vector model
def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])


# Create a new soup feature
metadata['soup'] = metadata.apply(create_soup, axis=1)

# Create a Count Vectorizer object (We won't use TF IDF because we do not want to down-weight the actor/director's
# presence if he or she has acted or directed in relatively more movies)
count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(metadata['soup'])

# Map the movie titles with their indices
metadata = metadata.reset_index()
indices2 = pd.Series(metadata.index, index=metadata['title']).drop_duplicates()


# A Function that takes in movie title and cos_sim (1 or 2 based on the plot or taking into account the actors,
# director, genre and keywords) as input and outputs most similar movies
def GetRecommendations(title, cos_sim):
    # Based on Plot of the last movie the user watched
    if cos_sim == 1:
        # Get the index of the movie that matches the title
        idx = indices1[title]

        # Compute the cosine similarity matrix
        cosine_sim1 = linear_kernel(tfidf_matrix[idx], tfidf_matrix)

        # Convert it into a list of tuples where the 1st element is its position, and the 2nd is the similarity score.
        sim_scores = list(enumerate(cosine_sim1[0]))

    # Based on the Actors, Director, Genre and Keywords of the last movie the user watched
    elif cos_sim == 2:
        # Get the index of the movie that matches the title
        idx = indices2[title]

        # Compute the cosine similarity matrix of that movie
        cosine_sim2 = cosine_similarity(count_matrix[idx], count_matrix)

        # Convert it into a list of tuples where the 1st element is its position, and the 2nd is the similarity score.
        sim_scores = list(enumerate(cosine_sim2[0]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    if cos_sim == 1:
        return df['title'].iloc[movie_indices].drop_duplicates()
    elif cos_sim == 2:
        return metadata['title'].iloc[movie_indices].drop_duplicates()


print()
df_new = GetRecommendations(input('Enter the movie that you last watched: '), int(input('1. Plot based recommendation'
                                                                                        '\n2. Actors, Director, Genre '
                                                                                        'and Keyword based'
                                                                                        ' recommendation\nEnter '
                                                                                        'your choice: ')))
df_new.index = np.arange(1, len(df_new) + 1)
print()
print("Top 10 movies recommended for you based on the the last movie you watched: ")
print()
print(df_new)
