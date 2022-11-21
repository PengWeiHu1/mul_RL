import os
import torch
import numpy as np
import pandas as pd
from tqdm import trange
from utils import to_vocab,model
from stackrnn import StackAugmentedRNN
from single_reinforce import Reinforcement
from RL_utils import Predictor,get_reward_pIC,q,estimate_and_update,get_reward_logP,search_thresh
path1 = os.getcwd()
vocab=to_vocab()
vocab.sort()
vocab_size=len(vocab)
ou2='output/checkpoint_biggest_rnn'
ou1='output/ligand.pkl'
use_cuda = torch.cuda.is_available()
prior=model(vocab_size,use_cuda)
prior.load_model(ou1)
agent=model(vocab_size,use_cuda)
agent.load_model(ou2)

my_Predictor=Predictor(path='datasets/ligand_226.csv',model_path="Exp8_Temperature_" +'\\'+'ECFP4'+'\\'+'ECFP4_phy_True_biao_True'+'_'+'model_')



def main(proper,agent,prior,my_Predictor,epsilon,out,out1,thresh_set=None,explore=True):
    memory_smile=[]
    if proper=='pIC':
        RL_max = Reinforcement(agent,prior, my_Predictor, get_reward_pIC,memory_smile,use_cuda)
    else:
        RL_max = Reinforcement(agent,prior, my_Predictor, get_reward_logP,memory_smile,use_cuda) 
    rewards_aver =[]
    rl_losses_aver = []
    epsilon1=[]
    n_ter=150
    for epoch in range(0,n_ter+1,1):
        for j in trange(15, desc='Policy gradient...'):
            cur_reward, cur_loss = RL_max.policy_gradient(epsilon)
            if explore:
                if len(rewards_aver) >2:
                    epsilon = search_thresh(rewards_aver[-3:],thresh_set)
               
                epsilon1.append(epsilon)  
            rewards_aver.append(q(rewards_aver, cur_reward)) 
            rl_losses_aver.append(q(rl_losses_aver, cur_loss))
    
        smiles_list1,val_perc1,uniq1,int_div1,pred_logp1,logp_valid1,pred_qed1,pred_sa1,pred_pic1,pic_valid1 = estimate_and_update(RL_max.generator,my_Predictor,use_cuda,vocab)
   
        values= [epoch,val_perc1,uniq1,int_div1,pred_logp1,logp_valid1,pred_qed1,pred_sa1,pred_pic1,pic_valid1]  
        if not os.path.exists(out):
            file=[]
            file.to_csv(out,header=None,index=None)
       
        file=[k.rstrip().split(',') for k in open(out).readlines()]
        file.append(values)
        file=pd.DataFrame(file)
        file.to_csv(out,header=None,index=None)
        if epoch % 5==0:  
          #  filename='stack_reinforce_pic_mo_all_weight_'+str(epsilon)+'_'+str(step)
              torch.save(RL_max.generator.state_dict(), path1+'//'+out1+'//'+'stackGRU'+str(epoch)+'.pkl')
        print('Sample trajectories:')
        
if __name__ == '__main__':

    epsilon=0.001 # 0.0001,0.01,0.05

    
    propers=['pIC','logP']
    explores=[True,False]
    for proper in propers:
        for explore in explores:
            if explore:
                thresh_set=1#2,3              
                out=proper+str(thresh_set)+'.csv'
                out1=proper+str(thresh_set)
                main(proper,agent,prior,my_Predictor,epsilon,out,out1,thresh_set,explore)                 
            else:
                thresh_set=None


                out=proper+str(epsilon)+'.csv'
                out1=proper+str(epsilon)
                main(proper,agent,prior,my_Predictor,epsilon,out,out1,thresh_set,explore)



























































