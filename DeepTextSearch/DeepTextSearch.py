import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pickle
import os


corpus_list_data = os.path.join('embedding-data/','corpus_list_data.pickle')
corpus_embeddings_data = os.path.join('embedding-data/','corpus_embeddings_data.pickle')

class LoadData:
    def __init__(self):
        self.corpus_list = None
    def from_csv(self,file_path:str):
        self.file_path = file_path
        csv_data = pd.read_csv(file_path)
        column_name = str(input('Input the text Column Name Please ? : '))
        self.corpus_list =  csv_data[column_name].dropna().to_list()
        return self.corpus_list

class TextEmbedder:
    def __init__(self):
        self.corpus_embeddings_data = corpus_embeddings_data
        self.corpus_list_data = corpus_list_data
        self.corpus_list = None
        self.embedder = SentenceTransformer('paraphrase-xlm-r-multilingual-v1')
        self.corpus_embeddings = None
        if 'embedding-data' not in os.listdir():
            os.makedirs("embedding-data")
    def embed(self,corpus_list:list):
        self.corpus_list = corpus_list
        if len(os.listdir("embedding-data/"))==0:
            self.corpus_embeddings = self.embedder.encode(self.corpus_list, convert_to_tensor=True,show_progress_bar=True)
            pickle.dump(self.corpus_embeddings, open(self.corpus_embeddings_data, "wb"))
            pickle.dump(self.corpus_list, open(self.corpus_list_data, "wb"))
            print("Embedding data Saved Successfully!")
            print(os.listdir("embedding-data/"))
        else:
            print("Embedding data allready present, Do you want Embed & Save Again? Enter yes or no")
            flag  = str(input())
            if flag.lower() == 'yes':
                self.corpus_embeddings = self.embedder.encode(self.corpus_list, convert_to_tensor=True,show_progress_bar=True)
                #np.savez(self.corpus_embeddings_data,self.corpus_embeddings.cpu().data.numpy())
                #np.savez(self.corpus_list_data,self.corpus_list)
                pickle.dump(self.corpus_embeddings, open(self.corpus_embeddings_data, "wb"))
                pickle.dump(self.corpus_list, open(self.corpus_list_data, "wb"))
                print("Embedding data Saved Successfully Again!")
                print(os.listdir("embedding-data/"))
            else:
                print("Embedding data allready Present, Please Apply Search!")
                print(os.listdir("embedding-data/"))
    def load_embedding(self):
        if len(os.listdir("embedding-data/"))==0:
            print("Embedding data Not present, Please Run Embedding First")
        else:
            print("Embedding data Loaded Successfully!")
            print(os.listdir("embedding-data/"))
            return pickle.load(open(self.corpus_embeddings_data, "rb"))

class TextSearch:
    def __init__(self):
        self.corpus_embeddings = pickle.load(open(corpus_embeddings_data, "rb"))
        self.data = pickle.load(open(corpus_list_data, "rb"))
    def find_similar(self,query_text:str,top_n=10):
        self.top_n = top_n
        self.query_text = query_text
        self.query_embedding = TextEmbedder().embedder.encode(self.query_text, convert_to_tensor=True)
        self.cos_scores = util.pytorch_cos_sim(self.query_embedding, self.corpus_embeddings)[0].cpu().data.numpy()
        self.sort_list  = np.argsort(-self.cos_scores)
        self.all_data  = []
        for idx in self.sort_list[1:self.top_n+1]:
            data_out = {}
            data_out['index'] = int(idx)
            data_out['text'] = self.data[idx]
            data_out['score'] = self.cos_scores[idx]
            self.all_data.append(data_out)
        return self.all_data