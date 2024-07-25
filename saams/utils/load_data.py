import numpy as np
import scipy.sparse as sp
import torch
import pickle
import torch.nn.functional as F
import os
import anndata as ad

def load_cell_embeddings(adatas_path):
    adatas_list = [i for i in os.listdir(adatas_path) if i.endswith('.h5ad')]
    for adatas_li in adatas_list:
        adata = ad.read(adatas_path + adatas_li)
        rows, cols = adata.X.nonzero()

        values = adata.X.data
    pass



def load_data():
    
    pass