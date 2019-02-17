import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def get_title_from_index(index):
    return df[df.index == index]["title"].values[0]


def get_index_from_title(title):
    return df[df.title == title]["index"].values[0]


# Reading CSV File
df = pd.read_csv("movie_dataset_content.csv", encoding='utf-8')


# Selecting Features
features = ['keywords', 'cast', 'genres', 'director']

# Creating a column in DF which combines all selected features
for feature in features:
    df[feature] = df[feature].fillna('')


def combine_features(row):
    return row['keywords'] + " " + row['cast'] + " " + row["genres"] + " " + row["director"]


df["combined_features"] = df.apply(combine_features, axis=1)

# making an object of CountVectorizer class to create count matrix
cv = CountVectorizer()

# Creating count matrix from this new combined column
count_matrix = cv.fit_transform(df["combined_features"])

# Computing the Cosine Similarity based on the count_matrix
cosine_sim = cosine_similarity(count_matrix)

movie_liked_by_user = "Thor"

# Getting index of this movie from its title
liked_movie_index = get_index_from_title(movie_liked_by_user)

similar_movies = list(enumerate(cosine_sim[liked_movie_index]))

# Get a list of similar movies in descending order of similarity score
predictions = sorted(similar_movies, key=lambda x: x[1], reverse=True)

# Print titles of 10 predicted movies
i = 0
for movie in predictions:
    print(get_title_from_index(movie[0]))
    i = i+1
    if i > 10:
        break
