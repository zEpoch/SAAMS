import saams
import pickle

gene_sequence_embedding = saams.utils.preprocess.get_gene_sequence_embedding_from_csv('data/processed/sequence.csv', 'gene_id', 'gene_sequence')