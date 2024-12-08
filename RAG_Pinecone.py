from datasets import load_dataset
import pandas as pd
data = load_dataset("Hessa/tqa_all_topics", split="train")
tqa_all_topics_df = pd.DataFrame(data)

# Dataset({
#     features: ['id', 'text', 'metadata'],
#     num_rows: 5001
# })

import os
import openai
import getpass  # platform.openai.com

# get API key from top-right dropdown on OpenAI website
openai.api_key = os.getenv("OPENAI_API_KEY") or getpass.getpass("Enter your OpenAI API key: ")

import pinecone

# initialize connection to pinecone (get API key at app.pinecone.io)
api_key = os.getenv("PINECONE_API_KEY") or getpass.getpass()
# find your environment next to the api key in pinecone console
env = os.getenv("PINECONE_ENVIRONMENT") or input()

pinecone.init(api_key=api_key, environment=env)

import time

index_name = "tqatopics"

# check if index already exists (it shouldn't if this is first time)
if index_name not in pinecone.list_indexes():
    # if does not exist, create index
    pinecone.create_index(
        index_name,
        dimension=1536,  # dimensionality of ada 002
        metric='dotproduct'
    )
    # wait for index to be initialized
    while not pinecone.describe_index(index_name).status['ready']:
        time.sleep(1)

# connect to index
index = pinecone.Index(index_name)
time.sleep(1)
# view index stats
index.describe_index_stats()

# {'dimension': 1536,
#  'index_fullness': 0.05001,
#  'namespaces': {'': {'vector_count': 5001}},
#  'total_vector_count': 5001}

embed_model = "text-embedding-ada-002"

from tqdm.auto import tqdm

batch_size = 100  # how many embeddings we create and insert at once

for i in tqdm(range(0, len(data), batch_size)):
    passed = False
    # find end of batch
    i_end = min(len(data), i+batch_size)
    # create batch
    batch = data[i:i_end]
    # create embeddings (exponential backoff to avoid RateLimitError)
    for j in range(5):  # max 5 retries
        try:
            res = openai.Embedding.create(input=batch["text"], engine=embed_model)
            passed = True
        except openai.error.RateLimitError:
            time.sleep(2**j)  # wait 2^j seconds before retrying
            print("Retrying...")
    if not passed:
        raise RuntimeError("Failed to create embeddings.")
    # get embeddings
    embeds = [record['embedding'] for record in res['data']]
    to_upsert = list(zip(batch["id"], embeds, batch["metadata"]))
    # upsert to Pinecone
    index.upsert(vectors=to_upsert)

import nltk
from nltk.tokenize import word_tokenize

# Download the Punkt tokenizer models (if not already downloaded)
nltk.download('punkt')

def count_tokens(text):
    tokens = word_tokenize(text)
    return len(tokens)

def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

def get_docs(query: str, top_k: int):
    xq = get_embedding(query)
    res = index.query(xq, top_k=top_k, include_metadata=True)
    # get doc text
    docs = {x["metadata"]['text']: i for i, x in enumerate(res["matches"])}
    # docs = {f"{x['id']} - {x['metadata']['lesson']} - {x['metadata']['text']}": i for i, x in enumerate(res["matches"])}
    return docs


def get_topic_id(query: str, top_k: int):
    xq = get_embedding(query)
    res = index.query(xq, top_k=top_k, include_metadata=True)
    id = {x['id']: i for i, x in enumerate(res["matches"])}
    return id

def get_all_context(docs):
  allText = ""
  for key, value in docs.items():
        allText += key
  return allText

def compare(query: str, top_k: int, top_n: int):
    # first get vec search results
    docs = get_docs(query, top_k=top_k)
    i2doc = {docs[doc]: doc for doc in docs.keys()}
    # rerank
    rerank_docs = co.rerank(
        query=query, documents=docs.keys(), top_n=top_n, model="rerank-english-v2.0"
    )
    original_docs = []
    reranked_docs = []
    # compare order change
    for i, doc in enumerate(rerank_docs):
        rerank_i = docs[doc.document["text"]]
        print(str(i)+"\t->\t"+str(rerank_i))
        if i != rerank_i:
            reranked_docs.append(f"[{rerank_i}]\n"+doc.document["text"])
            original_docs.append(f"[{i}]\n"+i2doc[i])
    for orig, rerank in zip(original_docs, reranked_docs):
        print("ORIGINAL:\n"+orig+"\n\nRERANKED:\n"+rerank+"\n\n---\n")

  
