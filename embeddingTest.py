from transformers import AutoTokenizer, AutoModel
import torch
import requests

response = requests.get("https://www.govinfo.gov/content/pkg/PLAW-115publ334/html/PLAW-115publ334.htm")
data = response.text

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


# Sentences we want sentence embeddings for
query = "What provisions are there to help organic farmers?"

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/multi-qa-mpnet-base-dot-v1")
model = AutoModel.from_pretrained("sentence-transformers/multi-qa-mpnet-base-dot-v1")

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