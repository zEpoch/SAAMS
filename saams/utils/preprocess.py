import pandas as pd
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import numpy as np
import pickle
import protfasta 
import csv 

def get_gene_sequence_csv_from_fasta(file_path: str,
                                     save_path: str):
    '''
    file_path: str
        Path to fasta file
    save_path: str
        Path to save csv file
    
    '''
    sequences = protfasta.read_fasta(file_path)
    sequences_list = []
    for sequence_id in sequences:
        sequence_dic = {'sequence_id': sequence_id, 'sequence': sequences[sequence_id]}
        
        sequences_list.append(sequence_dic)
    field_names = ['sequence_id', 'sequence'] 
    with open(save_path, 'w') as csvfile: 
        writer = csv.DictWriter(csvfile, fieldnames = save_path) 
        writer.writeheader() 
        writer.writerows(sequences_list) 


def get_gene_sequence_embedding_from_csv(file_path: str, 
                                         gene_id: str, 
                                         gene_sequence: str):
    '''
    file_path: str
        Path to csv file
    gene_id: str
        Column name of gene id
    gene_sequence: str
        Column name of gene sequence
    '''
    csv = pd.read_csv(file_path)
    length = len(csv[gene_id])
    gene_id_list = csv[gene_id].tolist()
    gene_sequence_list = csv[gene_sequence].tolist()
    gene_sequence_list_embedding = get_sequence_embedding(gene_sequence_list)
    
    dic = {}
    
    for gene_id, gene_sequence in zip(gene_id_list, gene_sequence_list_embedding):
        dic[gene_id] = gene_sequence
    
    pickle.dump(dic, open('dic.pkl', 'wb'))
    return dic

def get_sequence_embedding(sequence_list: list, 
                           model_name: str = "InstaDeepAI/nucleotide-transformer-2.5b-multi-species"):
    '''
    sequence_list: list
        List of sequences
    model_name: str
        Model name of the transformer
    '''
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    tokens_ids = tokenizer.batch_encode_plus(sequence_list, return_tensors="pt", padding = True)["input_ids"]

    # Compute the embeddings
    attention_mask = tokens_ids != tokenizer.pad_token_id #type: bool
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
    # pickle.dump(attention_mask, open('attention_mask.pkl', 'wb'))
    # return embeddings
    # Compute mean embeddings per sequence
    # mean_sequence_embeddings = torch.sum(attention_mask.unsqueeze(-1)*embeddings, axis=-2) / torch.sum(attention_mask.shape[1], axis=-1) # type: ignore
    
    embeddings = torch.Tensor(embeddings)
    attention_mask = attention_mask.unsqueeze(-1) # type: ignore
    
    attention_mask = torch.Tensor(attention_mask)  # type: ignore # Convert attention_mask to torch.Tensor
    
    mean_sequence_embeddings = torch.sum(attention_mask*embeddings, axis=-2) / attention_mask.shape[1]  # type: ignore
    print(f"Mean sequence embeddings: {mean_sequence_embeddings}")
    pickle.dump(mean_sequence_embeddings, open('mean_sequence_embeddings.pkl', 'wb'))
    return mean_sequence_embeddings

