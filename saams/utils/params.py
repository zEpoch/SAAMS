import argparse
import sys
def params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pt_emb_path', type = str, help = 'protein embedding path')
    

    args, _ = parser.parse_known_args()
    return args

def set_params():
    
    args = params()
    return args