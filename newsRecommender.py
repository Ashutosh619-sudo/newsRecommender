import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer


df = pd.read_json("News_Category_Dataset_v2.json",lines=True)
df.head(5)

df = df[["category","headline","short_description"]]

df = df.head(2000)
df.head()

for index,row in df.iterrows():
    row["category"] = row["category"].lower()

df["full_news"] = ""

for index, row in df.iterrows():
    df["full_news"] = df["category"]+" "+df["headline"]+df["short_description"]

df = df[["full_news","headline"]]


df["full_news"] = df["full_news"].map(lambda x:x.lower())
df["headline"] = df["headline"].map(lambda x:x.lower())

df.set_index("headline",inplace=True)


token = RegexpTokenizer(r'[a-zA-Z0-9]+')
count = CountVectorizer(lowercase = True,stop_words="english",tokenizer=token.tokenize)
count_matrix = count.fit_transform(df["full_news"])

indices = pd.Series(df.index)

cosine_sim = cosine_similarity(count_matrix,count_matrix)

def recommendations(headline,cosine_sim=cosine_sim):
    recommended_movies = []

    idx = indices[indices==headline].index[0]

    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending=False)

    top = list(score_series.iloc[1:11].index)

    for i in top:
        recommended_movies.append(list(df.index)[i])
        
    return recommended_movies


print(indices[0])
print(recommendations(indices[0]))
    
