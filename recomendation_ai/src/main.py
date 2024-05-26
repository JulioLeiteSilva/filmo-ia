import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import sys
import json

def recomendacao(nomeFilme):

    recom_data = pd.read_csv("../filmo-ia/recomendation_ai/data/processed/recom_data_processed.csv") 
    recom_data['index'] = range(0, len(recom_data))
    recom_data = recom_data.set_index('index').reset_index()
    
    selected_features = ['genre','crew',"orig_lang"]
    recom_data.fillna('', inplace=True)
    
    combined_features = recom_data['genre']+' '+recom_data['overview']+' '+recom_data['crew']
    vectorizer = TfidfVectorizer(stop_words='english')
    feature_vectors = vectorizer.fit_transform(combined_features)
    
    similarity = cosine_similarity(feature_vectors)
    
    list_of_all_titles = recom_data['names'].tolist()
    find_close_match = difflib.get_close_matches(nomeFilme, list_of_all_titles)
    if not find_close_match:
        return "Nenhuma correspondência encontrada para o nome do filme fornecido."
    close_match = find_close_match[0]
    index_of_the_movie = recom_data[recom_data.names == close_match]['index'].values[0]
    similarity_score = list(enumerate(similarity[index_of_the_movie]))
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)
    
    MoviesArray = [recom_data.iloc[movie[0]]['names'] for movie in sorted_similar_movies[1:70]]

    return MoviesArray

print(json.dumps(recomendacao(str(sys.argv[1]))))