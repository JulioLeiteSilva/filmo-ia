import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import sys
import timeit

def recomendacao(nomeFilme):
    initTime = timeit.default_timer()
    start = timeit.default_timer()
    recom_data = pd.read_csv("../filmo-ia/recomendation_ai/data/processed/recom_data_processed.csv")
    print(f"Leitura do CSV: {timeit.default_timer() - start} segundos")
    
    start = timeit.default_timer()
    recom_data['index'] = range(0, len(recom_data))
    recom_data = recom_data.set_index('index').reset_index()
    print(f"Preparo dos dados: {timeit.default_timer() - start} segundos")
    
    start = timeit.default_timer()
    selected_features = ['genre','crew',"orig_lang"]
    recom_data.fillna('', inplace=True)
    print(f"Preenchimento de valores nulos: {timeit.default_timer() - start} segundos")
    
    start = timeit.default_timer()
    combined_features = recom_data['genre']+' '+recom_data['overview']+' '+recom_data['crew']
    vectorizer = TfidfVectorizer(stop_words='english')
    feature_vectors = vectorizer.fit_transform(combined_features)
    print(f"Transformação TF-IDF: {timeit.default_timer() - start} segundos")
    
    start = timeit.default_timer()
    similarity = cosine_similarity(feature_vectors)
    print(f"Cálculo da similaridade: {timeit.default_timer() - start} segundos")
    
    start = timeit.default_timer()
    list_of_all_titles = recom_data['names'].tolist()
    find_close_match = difflib.get_close_matches(nomeFilme, list_of_all_titles)
    if not find_close_match:
        return "Nenhuma correspondência encontrada para o nome do filme fornecido."
    close_match = find_close_match[0]
    index_of_the_movie = recom_data[recom_data.names == close_match]['index'].values[0]
    similarity_score = list(enumerate(similarity[index_of_the_movie]))
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)
    print(f"Busca de correspondências e ordenação: {timeit.default_timer() - start} segundos")
    
    start = timeit.default_timer()
    i = 1
    MoviesArray = [recom_data.iloc[movie[0]]['names'] for movie in sorted_similar_movies[1:40]]
    print(f"Construção do array de filmes: {timeit.default_timer() - start} segundos")
    print(f"Tempo de Tudo: {timeit.default_timer() - initTime} segundos")
    
    return MoviesArray

print(recomendacao("Iron man"))
#print(recomendacao(str(sys.argv[1])))