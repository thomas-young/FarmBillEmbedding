from sentence_transformers import SentenceTransformer, util
import torch
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import re
min_sentence_length = 200
response = requests.get("https://www.govinfo.gov/content/pkg/PLAW-115publ334/html/PLAW-115publ334.htm")
data = response.text
soup = BeautifulSoup(data, 'html.parser')
data = soup.get_text()


# Split the text into sentences using regular expressions
#data = re.split('(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', data)

data = re.split(r'\n\s*\n', data)

# Remove sentences that are shorter than the specified minimum length
data = [re.sub('\s+', ' ', s.strip().replace('\n', '')) for s in data if len(s) >= min_sentence_length]
for paragraph in data:
	print(paragraph)
	print()
	print()
embedder = SentenceTransformer('multi-qa-mpnet-base-dot-v1')

def encode_sentences(transcripts, batch_size=256):

    # loop through in batches of 64
    all_batches = []
    for i in range(0, len(transcripts), batch_size):
        # find end position of batch (for when we hit end of data)
        i_end = min(len(transcripts), i + batch_size)

        batch_text = transcripts[i:i_end]
        print(batch_text)
        # create the embedding vectors
        batch_vectors = embedder.encode(batch_text, convert_to_tensor=True).tolist()

        # add batch to list of all batches
        all_batches.append(batch_vectors)

    return all_batches

# Corpus with example sentences
corpus = ['A man is eating food.',
          'A man is eating a piece of bread.',
          'The girl is carrying a baby.',
          'A man is riding a horse.',
          'A woman is playing violin.',
          'Two men pushed carts through the woods.',
          'A man is riding a white horse on an enclosed ground.',
          'A monkey is playing drums.',
          'A cheetah is running behind its prey.'
          ]
#corpus_embeddings = embedder.encode(data, convert_to_tensor=True)
corpus_embeddings = encode_sentences(data)

# Query sentences:
queries = ['What are programs that can help organic farmers?']


# Find the closest 15 sentences of the corpus for each query sentence based on cosine similarity
top_k = min(15, len(corpus))
for query in queries:
    query_embedding = embedder.encode(query, convert_to_tensor=True)

    # We use cosine-similarity and torch.topk to find the highest 5 scores
    cos_scores = torch.Tensor()
    for corpus_embedding in corpus_embeddings:
        cos_scores_temp = util.cos_sim(query_embedding, corpus_embedding)[0]
        cos_scores = torch.cat((cos_scores, cos_scores_temp), 0)

        # print cos_score type
        print(type(cos_scores))
        # print cos_score dimensions
        print(cos_scores.shape)
    top_results = torch.topk(cos_scores, k=top_k)

    print("\n\n======================\n\n")
    print("Query:", query)
    print("\nTop 15 most similar sentences in corpus:")

    for score, idx in zip(top_results[0], top_results[1]):
        print(data[idx], "(Score: {:.4f})".format(score))


