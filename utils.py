import os
import gzip
import time
import math
import torch
import random
from tqdm import tqdm
from rdkit import Chem
from tqdm import trange
from rdkit import DataStructs
from stackrnn import StackAugmentedRNN
from rdkit.Chem import Descriptors as desc
from rdkit.Chem import Descriptors,MolFromSmiles

def jieya(file):
    gz_file = gzip.open(file)
    suppl = Chem.ForwardSDMolSupplier(gz_file)
    mols=[]
    for mol in suppl:    
        try:
            mols.append(Chem.MolToSmiles(mol,isomericSmiles=False))     
        except:
            print(mol)
    return mols

def remove_duplicates(smiles_list):
    unique_set = set()
    unique_list = []
    for element in smiles_list:
        if element not in unique_set:
            unique_set.add(element)
            unique_list.append(element)
            
    return unique_list,(len(unique_list)/len(smiles_list))*100

def logp_mw(smiles):
    smi=[]
    p=[]
    mw1=[]
    for smile in smiles:   
        try:
            res_molecule = MolFromSmiles(smile)
        except Exception:
            res_molecule=None

        if res_molecule==None:
            logP= -666
        else:
            logP=Descriptors.MolLogP(res_molecule)
       
            mw=desc.MolWt(res_molecule)
            p.append(logP)
            mw1.append(mw)
        if -2 <= logP <= 6 and  200 <= mw <= 600:
            smi.append(smile)
    return smi

def read_csv(file):
    curr_path = os.getcwd()          
    f_mol = ''                                        #with open(path_generator + 'generatedMol.txt', 'r') as f:
    with open(curr_path + "\\"+file, 'r') as f:#smiles29
        for line in f:
            f_mol = f_mol + line
    generated_smiles = f_mol.split('\n')[:-1]
    
    return generated_smiles

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def to_vocab():
    vocab=['G', '$', '#', '%', ')', '(', '+', '-', '/', '.', '1', '0', '3', '2', '5', '4', '7',
              '6', '9', '8', '=', 'A', '@', 'C', 'B', 'F', 'I', 'H', 'O', 'N', 'P', 'S', '[', ']',
              '\\', 'c', 'e', 'i', 'l', 'o', 'n', 'p', 's', 'r', '\n']
    return vocab

def add_tok_tihuang(smiles):
    file = []
    for smile in tqdm(smiles):
        smile = smile.replace('Cl', 'L').replace('Br', 'R')
        file.append('G' + smile + '$') 
    return file


def model(vocab_size,use_cuda):

    hidden_size = 1500
    stack_width = 1500
    stack_depth = 200
    layer_type = 'GRU'
    lr = 0.001
    optimizer_instance = torch.optim.Adadelta

    my_generator1 = StackAugmentedRNN(input_size=vocab_size, hidden_size=hidden_size,
                                     output_size=vocab_size, layer_type=layer_type,
                                     n_layers=1, is_bidirectional=False, has_stack=True,
                                     stack_width=stack_width, stack_depth=stack_depth, 
                                     use_cuda=use_cuda, 
                                     optimizer_instance=optimizer_instance, lr=lr)
    return my_generator1


def char_tensor(string,use_cuda,vocab):
    tensor = torch.zeros(len(string)).long()
    
    for c in range(len(string)):
        tensor[c] = vocab.index(string[c])
    if use_cuda:
        return torch.tensor(tensor).cuda()
    else:      
        return torch.tensor(tensor)

def random_smile(file,use_cuda,vocab):
    index = random.randint(0, len(file)-1)
    smile=file[index]
    inp = char_tensor(smile[:-1],use_cuda,vocab)
    target = char_tensor(smile[1:],use_cuda,vocab)
    return inp,target

#%%
def sampleing(generation,use_cuda,vocab,sample=None, prime_str='G', end_token='$', predict_len=100):
    hidden = generation.init_hidden()
    if generation.has_cell:
        cell = generation.init_cell()
        hidden = (hidden, cell)
    if generation.has_stack:
        stack = generation.init_stack()
    else:
        stack = None
    prime_input = char_tensor(prime_str,use_cuda,vocab)
    new_sample = prime_str

    # Use priming string to "build up" hidden state
    for p in range(len(prime_str)-1):
        _, hidden, stack = generation.forward(prime_input[p], hidden, stack)
    inp = prime_input[-1]

    for p in range(predict_len):
        output, hidden, stack = generation.forward(inp, hidden, stack)
        if sample !=None:
            output=output/sample

        # Sample from the network as a multinomial distribution
        probs = torch.softmax(output, dim=1)
        top_i = torch.multinomial(probs.view(-1), 1)[0].cuda()#.numpy()

        # Add predicted character to string and use as next input
        predicted_char = vocab[top_i]
        new_sample += predicted_char
        inp = char_tensor(predicted_char,use_cuda,vocab)
        if predicted_char == end_token:
            break
    return new_sample
def fit(generation, data,use_cuda,vocab, n_iterations, out,all_losses=[], print_every=10000,
        plot_every=10, augment=False):

    start = time.time()
    loss_avg = 0
    path = os.getcwd()
    
    for epoch in trange(1, n_iterations + 1, desc='Training in progress...'):
        inp, target = random_smile(data,use_cuda,vocab)
        loss = generation.train_step(inp, target)
        loss_avg += loss
        model_path=path+'//'+out
        if epoch % print_every == 0:
            print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch,
                                           epoch / n_iterations * 100, loss)
                  )
            print(sampleing(generation,use_cuda,vocab,sample=1), '\n')

        if epoch % plot_every == 0:
            all_losses.append(loss_avg / plot_every)
            loss_avg = 0
        if epoch % 50000==0:
            generation.save_model(model_path)
          
    return all_losses

def diversity(smiles_A,smiles_B = None):

 
    td = 0
    
    fps_A = []
    for i, row in enumerate(smiles_A):
        try:
            mol = Chem.MolFromSmiles(row)
            fps_A.append(Chem.AllChem.GetMorganFingerprint(mol, 3))
        except:
            'ERROR: Invalid SMILES!'
            
        
    
    if smiles_B == None:
        for ii in range(len(fps_A)):
            for xx in range(len(fps_A)):
                ts = 1 - DataStructs.TanimotoSimilarity(fps_A[ii], fps_A[xx])
                td += ts          
        
        if len(fps_A) == 0:
            td = None
        else:
            td = td/len(fps_A)**2
    else:
        fps_B = []
        for j, row in enumerate(smiles_B):
            try:
                mol = Chem.MolFromSmiles(row)
                fps_B.append(Chem.AllChem.GetMorganFingerprint(mol, 3))
            except:
                print('ERROR: Invalid SMILES!') 
        
        
        for jj in range(len(fps_A)):
            for xx in range(len(fps_B)):
                ts = 1 - DataStructs.TanimotoSimilarity(fps_A[jj], fps_B[xx]) 
                td += ts
        
        if (len(fps_A) == 0 or len(fps_B) == 0):
            td = None
        else:   
            td = td / (len(fps_A)*len(fps_B))
    
    return td

def validity(smiles_list):

    
    total = len(smiles_list)
    valid_smiles =[]
    count = 0
    for sm in smiles_list:
        if Chem.MolFromSmiles(sm) != None and sm !='':
#        if MolFromSmiles(sm,sanitize=False) != None and sm !='':
            valid_smiles.append(sm)
            count = count +1
    perc_valid = count/total*100
    
    return valid_smiles, perc_valid


