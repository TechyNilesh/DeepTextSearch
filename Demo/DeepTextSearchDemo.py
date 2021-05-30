# Importing the proper classes
from DeepTextSearch import LoadData,TextEmbedder,TextSearch

# Load data from CSV file
data = LoadData().from_csv("../your_file_name.csv")

# For Serching we need to Embed Data first, After Embedding all the data stored on the local path
TextEmbedder().embed(corpus_list=data)

# for searching, you need to give the query_text  and the number of the similar text you want
TextSearch().find_similar(query_text="What are the key features of Node.js?",top_n=10)

