import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/msmarco-distilbert-dot-v5')


sentences = ['i want to kill myself', 'i want to die', 'holeless','i am depressed','i want to commit suicide']


sentence_embeddings = model.encode(sentences)


def cosine(u, v):
    '''
    Computes Similarity between two seeences

    '''
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

def degreee_suicidal(query):
    '''
    Input: Take in a sentence:
    Output: Matches it against the each of the
    sentences = ['i want to kill myself', 'i want to die', 'holeless','i am depressed','i want to commit suicide']
    and returns the 1/max(most_similar_value)
    '''
    emotion=[]
    query_vec=model.encode([query])[0]

    for sent in sentences:
        sim = cosine(query_vec, model.encode([sent])[0])
        emotion.append(sim)
        return 1/(max(emotion))
        #print("Sentence = ", sent, "; similarity = ", sim)

#saving and creating files
analysing_title_gro_merged=pd.read_csv('/Users/ankitkothari/Documents/gdrivre/UMD/MSML-641-NLP/msml_641_project_scripts/merged_degreee_test.csv')
analysing_title_gro_merged['suicidal_degree']=analysing_title_gro_merged['clean_title_tokens'].map(degreee_suicidal)
print(analysing_title_gro_merged)
analysing_title_gro_merged.to_csv('/Users/ankitkothari/Documents/gdrivre/UMD/MSML-641-NLP/msml_641_project_scripts/merged_degreee_encoded_test.csv')

#plotting graph
import plotly.express as px
fig = px.box(analysing_title_gro_merged, x="labels", y="suicidal_degree")
fig.show()
