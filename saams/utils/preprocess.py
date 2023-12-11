import pandas as pd
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import numpy as np

def read_gene_sequence(file_path):
    csv = pd.read_csv(file_path)
    length = len(csv['gene_name'])
    dic = {}
    for i in range(length):
        seq = csv['sequence'][i]
        embedding = get_sequence_embedding(seq)
        print(embedding)
        embedding = embedding[0]
        print(embedding)
        #return embedding
        dic[csv['gene_name'][i]] = embedding.tolist()

    return dic

def get_sequence_embedding(sequence: str):

    tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-multi-species")
    model = AutoModelForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-multi-species")
    tokens_ids = tokenizer.batch_encode_plus([sequence], return_tensors="pt", padding = True)["input_ids"]

    # Compute the embeddings
    attention_mask = tokens_ids != tokenizer.pad_token_id
    torch_outs = model(
        tokens_ids,
        attention_mask=attention_mask,
        encoder_attention_mask=attention_mask,
        output_hidden_states=True
    )

    # Compute sequences embeddings
    embeddings = torch_outs['hidden_states'][-1].detach().numpy()
    print(f"Embeddings shape: {embeddings.shape}")
    # print(f"Embeddings per token: {embeddings}")

    # Compute mean embeddings per sequence
    mean_sequence_embeddings = torch.sum(attention_mask.unsqueeze(-1)*embeddings, axis=-2)/torch.sum(attention_mask, axis=-1)
    print(f"Mean sequence embeddings: {mean_sequence_embeddings}")
    return mean_sequence_embeddings

