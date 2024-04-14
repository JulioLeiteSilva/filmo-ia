import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

recom_data = pd.read_csv("data/processed/recom_data_processed.csv")

recom_data['index'] = range(0, len(recom_data))

recom_data = recom_data.set_index('index').reset_index()

selected_features = ['genre','crew',"orig_lang"]
print(selected_features)

for feature in selected_features:
    recom_data[feature] = recom_data[feature].fillna('')

combined_features = recom_data['genre']+' '+recom_data['overview']+' '+recom_data['crew']

vectorizer = TfidfVectorizer()

feature_vectors = vectorizer.fit_transform(combined_features)

similarity = cosine_similarity(feature_vectors)

movie_name = "The godfather"
print(f'Enter your favourite movie name : {movie_name}')

list_of_all_titles = recom_data['names'].tolist()

find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

print(" similiar names  :  ",  find_close_match ,"\n\n")

close_match = find_close_match[0]

index_of_the_movie = recom_data[recom_data.names == close_match]['index'].values[0]

similarity_score = list(enumerate(similarity[index_of_the_movie]))

sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True)

print('Movies suggested for you : \n')

i = 1

for movie in sorted_similar_movies:
    index = movie[0]
    title_from_index = recom_data[recom_data.index==index]['names'].values[0]
    if (i<100):
        print(i, '.',title_from_index)
        i+=1 