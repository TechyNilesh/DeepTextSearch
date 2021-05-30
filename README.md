# Deep Text Search - AI Based Text Search & Recommendation System
<p align="center"><img src="https://github.com/TechyNilesh/DeepTextSearch/blob/main/logo/DeepTextSearch%20Logo-2.png?raw=true" alt="Brain+Machine" height="218" width="350"></p>

**Deep Text Search** is an AI-powered multilingual **text search and recommendation engine** with state-of-the-art transformer-based **multilingual text embedding (50+ languages)**.

![Generic badge](https://img.shields.io/badge/DeepTextSerach-v1-orange.svg) ![Generic badge](https://img.shields.io/badge/Artificial_Intelligence-Advance-green.svg) ![Generic badge](https://img.shields.io/badge/Python-v3-blue.svg) ![Generic badge](https://img.shields.io/badge/pip-v3-red.svg)  ![Generic badge](https://img.shields.io/badge/SentenceTransformer-v1-orange.svg)

<h2><img src="https://cdn2.iconfinder.com/data/icons/artificial-intelligence-6/64/ArtificialIntelligence9-512.png" alt="Brain+Machine" height="38" width="38"> Creators </h2>

### [Nilesh Verma](https://nileshverma.com "Nilesh Verma")

## Features
- Faster Search.
- High Accurate Text Recommendation and Search Output Result.
- Best for Implementing on python based web application or APIs.
- Best implementation for College students and freshers for project creation.
- Applications are Text-based News, Social media post, E-commerce Product recommendation and other text-based platforms that want to implement text recommendation and search.

## Installation

This library is compatible with both *windows* and *Linux system* you can just use **PIP command** to install this library on your system:

```shell
pip install DeepTextSearch
```

## How To Use?

We have provided the **Demo** folder under the *GitHub repository*, you can find the example in both **.py** and **.ipynb**  file. Following are the ideal flow of the code:

#### 1. Importing the Important Classes
There are three important classes you need to load **LoadData** - for data loading, **TextEmbedder** - for embedding the text  to data, **TextSearch** - For searching the text.

```python
# Importing the proper classes
from DeepTextSearch import LoadData,TextEmbedder,TextSearch
```

#### 2. Loading the Images Data

For loading the images data we need to use the **LoadData** object, from there we can import text data as python list object from the CSV/Text  file.

```python
# Load data from CSV file
data = LoadData().from_csv("../your_file_name.csv")
# Load data from Text file
data = LoadData().from_text("../your_file_name.txt")
```
### 3. Embedding and Saving The File in Local Folder

For Embedding we are using state of the art multilingual Sentence Transformer Embedding, We also store the information of the Embedding for further use on the local path **[embedding-data/]** folder.

You can also use the **load embedding()** method in a **TextEmbedder()** class to load saved embedding data.

```python
# To use Serching, we must first embed data. After that, we must save all of the data on the local path.
TextEmbedder().embed(corpus_list=data)

# Loading Embedding data
corpus_embedding = TextEmbedder().load_embedding()
```
### 3. Searching

We compare Cosian Similarity for searching and recommending, and then the corpus is sorted according to the similarity score:

```python
# You must include the query text and the quantity of comparable texts you want to search for.
TextSearch().find_similar(query_text="What are the key features of Node.js?",top_n=10)
```

## Complete Code

```python
# Importing the proper classes
from DeepTextSearch import LoadData,TextEmbedder,TextSearch
# Load data from CSV file
data = LoadData().from_csv("../your_file_name.csv")
# To use Serching, we must first embed data. After that, we must save all of the data on the local path
TextEmbedder().embed(corpus_list=data)
# You must include the query text and the quantity of comparable texts you want to search for
TextSearch().find_similar(query_text="What are the key features of Node.js?",top_n=10)
```

## License

```rst
MIT License

Copyright (c) 2021 Nilesh Verma

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

**More cool features will be added in future. Feel free to give suggestions, report bugs and contribute.**
