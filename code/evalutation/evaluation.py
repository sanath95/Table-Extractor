from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import json
from tabulate import tabulate

def calculate_cosine_sim(string1, string2):
    # Convert the strings into vectors
    vectorizer = CountVectorizer().fit_transform([string1, string2])
    vectors = vectorizer.toarray()

    # Calculate the cosine similarity
    cosine_sim = cosine_similarity(vectors)

    # Extract the cosine similarity value between the two strings
    cosine_similarity_value = cosine_sim[0][1]
    return cosine_similarity_value


with open('./code/pipeline4/evalutation/results.txt', encoding='UTF-8') as f: 
    data = f.read() 

res = json.loads(data)

imgs = res.keys()
a_score = []
b_score = []
a_time = []
b_time = []
archetype = []

for k,v in res.items():
    a_score.append(calculate_cosine_sim(v['actual'], v['pred_pipeline_a']))
    b_score.append(calculate_cosine_sim(v['actual'], v['pred_pipeline_b']))
    a_time.append(v['time_pipeline_a'])
    b_time.append(v['time_pipeline_b'])

df = pd.DataFrame({
    "Image": imgs,
    "Pipeline A Cosine Similarity": a_score,
    "Pipeline B Cosine Similarity": b_score,
    "Pipeline A Time Taken (s)": a_time,
    "Pipeline B Time Taken (s)": b_time,
})

# print markdown string that can be displayed in the readme file
print(tabulate(df, tablefmt="pipe", headers="keys", showindex=False))


### TODO: calculate grouped average by archetype!!
# print(f'Average Pipeline A Cosine Similarity: {df.loc[:, "Pipeline A Cosine Similarity"].mean()}')
# print(f'Average Pipeline B Cosine Similarity: {df.loc[:, "Pipeline B Cosine Similarity"].mean()}')
# print(f'Average Pipeline A Time Taken (s): {df.loc[:, "Pipeline A Time Taken (s)"].mean()}')
# print(f'Average Pipeline B Time Taken (s): {df.loc[:, "Pipeline B Time Taken (s)"].mean()}')