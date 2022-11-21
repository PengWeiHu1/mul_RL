import os
import re
import math
import time
import torch
from tqdm import tqdm   
from tqdm import trange
from utils import read_csv,to_vocab,add_tok_tihuang,time_since,model,fit
os.environ['CUDA_ENABLE_DEVICES'] = '0'
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def Training(file_ligand):
    out1='output/checkpoint_biggest_rnn'
    out2='output/stack_ligand.pkl'
    

    vocab=to_vocab()
    
    vocab_size=len(vocab)
    use_cuda = torch.cuda.is_available()
    prior=model(vocab_size,use_cuda)
    '''if not os.path.exists(out1):
        generated_smiles=read_csv(file_chembl)
        file=add_tok_tihuang(generated_smiles)
        
        fit(prior,file,use_cuda,vocab,len(file),out1)'''
#%%
    prior.load_model(out1)
    ligand_smiles=read_csv(file_ligand)
    file1=add_tok_tihuang(ligand_smiles)
    fit(prior,file1,use_cuda,vocab,len(file1)*100,out2)

if __name__ == '__main__':
    
    file_ligand='datasets/ligand.txt'
    Training(file_ligand)
    
    
    








