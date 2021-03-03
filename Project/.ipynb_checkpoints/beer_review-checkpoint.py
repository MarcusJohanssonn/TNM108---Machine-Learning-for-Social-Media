import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df_beer = pd.read_csv('/Users/MarcusJohansson/Documents/Skola/TNM108/Projekt/beer_reviews3.csv', delimiter = ';')


#Helper function to retrieve the name of the beer from its index
def get_name_from_index(index):
 return df_beer["Beer"].iloc[index]
#Helper function to retrieve the index of the beer from the name
def get_index_from_name(name):
 return df_beer.index[df_beer['Beer'] == name]

def combine_features(row):
    try:
        return row["Aroma"] + " " + row["Appearance"] + " " + row["Taste"]+ " " + row["Palate"]+ " " + row["Overall"]+ " " + row["Comments"]+ " " + row["Country"]+ " " + row["Style"]
    except ValueError as err:
        pass
    
features = ["Aroma", "Appearance", "Taste", "Palate", "Overall","Comments","Country","Style"]

for feature in features:
 df_beer[feature] = df_beer[feature].fillna('')

df_beer["combined_features"] = df_beer.apply(combine_features,axis=1)

cv = CountVectorizer()
count_matrix = cv.fit_transform(df_beer["combined_features"])
cosine_sim = cosine_similarity(count_matrix)

beer_user_likes = "Nils Oscar Brown Ale"

#Getting the index of the beer in the Data Frame
beer_index = get_index_from_name(beer_user_likes)

similar_beers = list(enumerate(cosine_sim[beer_index[0]]))

#Sorting the beers in descending similarity order
sorted_similar_beers = sorted(similar_beers, key=lambda x:x[1],reverse=True)[1:]

for beer in sorted_similar_beers:
 print(get_name_from_index(beer[0]))
 