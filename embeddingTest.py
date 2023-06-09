from transformers import AutoTokenizer, AutoModel
import torch
import requests
from sentence_transformers import SentenceTransformer 
from tqdm import tqdm
response = requests.get("https://www.govinfo.gov/content/pkg/PLAW-115publ334/html/PLAW-115publ334.htm")
data = response.text

# remove html tags from data
from bs4 import BeautifulSoup
soup = BeautifulSoup(data, 'html.parser')
data = soup.get_text()

# count number of blank lines:
print(data.count('\n\n'))

# split the data at blank lines
data = data.split('\n\n')



# print length of data
print(len(data))
#print size of largest paragraph
print(max([len(d) for d in data]))

#CLS Pooling - Take output from first token
def cls_pooling(model_output):
    return model_output.last_hidden_state[:,0]

#Encode text
def encode(texts):
    # Tokenize sentences
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input, return_dict=True)

    # Perform pooling
    embeddings = cls_pooling(model_output)

    return embeddings

def encode_sentences(transcripts, batch_size=64):
    """
    Encoding all of our segments at once or storing them locally would require too much compute or memory.
    So we do it in batches of 64
    :param transcripts:
    :param batch_size:
    :return:
    """
    # loop through in batches of 64
    all_batches = []
    for i in tqdm(range(0, len(transcripts), batch_size)):
        # find end position of batch (for when we hit end of data)
        i_end = min(len(transcripts), i + batch_size)
        # extract the metadata like text, start/end positions, etc
        batch_meta = [{
            **row
        } for row in transcripts[i:i_end]]
        # extract only text to be encoded by embedding model
        batch_text = [
            row['text'] for row in batch_meta
        ]
        # create the embedding vectors
        batch_vectors = model.encode(batch_text).tolist()

        batch_details = [
            {
                **batch_meta[x],
                'vectors': batch_vectors[x]
            } for x in range(0, len(batch_meta))
        ]
        all_batches.extend(batch_details)

    return all_batches

# Sentences we want sentence embeddings for
query = "What provisions are there to help organic farmers?"

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/multi-qa-mpnet-base-dot-v1")
model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')

#Encode query and docs
query_emb = encode(query)
doc_emb = encode(data)

#Compute dot score between query and all document embeddings
scores = torch.mm(query_emb, doc_emb.transpose(0, 1))[0].cpu().tolist()

#Combine docs & scores
doc_score_pairs = list(zip(data, scores))

#Sort by decreasing score
doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)

#Output passages & scores
for doc, score in doc_score_pairs:
    print(score, doc)