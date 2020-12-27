import pandas as pd
df = pd.read_csv('indianexpress.csv')
documents = df['description'].to_list()
from sklearn.feature_extraction.text import CountVectorizer

count_vectorizer = CountVectorizer(stop_words='english')
count_vectorizer = CountVectorizer()
sparse_matrix = count_vectorizer.fit_transform(documents)

doc_term_matrix = sparse_matrix.todense()
df = pd.DataFrame(doc_term_matrix, 
                  columns=count_vectorizer.get_feature_names())
df
from tqdm import tqdm
import numpy as np

def dot(K, L):
   if len(K) != len(L):
      return 0
   return sum(i[0] * i[1] for i in zip(K, L))/(np.linalg.norm(K)*np.linalg.norm(L))

similar_score = np.zeros((30000, 6))
score = np.zeros((30000, 6))
for i in tqdm(range(0,30000)):
    y_true  = df.loc[i].to_numpy()
    temp = [0,0,0,0,0,0]
    #for best jaccard score
    temp1 = [0,0,0,0,0,0]
    for j in range(0,30000):
        if i==j:
            continue
        y_pred  = df.loc[j].to_numpy()
        temp2 = dot(y_true, y_pred)
        if temp2 > temp1[-1]:
            for k in range(6):
                if temp2 > temp1[k]:
                    temp1.insert(k,temp2)
                    temp1.pop()
                    temp.insert(k,j)
                    temp.pop()
    similar_score[i] = temp    
    score[i] = temp1