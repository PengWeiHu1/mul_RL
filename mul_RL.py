import os
import torch
import numpy as np
import pandas as pd
from tqdm import trange
from utils import to_vocab,model
from stackrnn import StackAugmentedRNN
from mul_reinforce import Reinforcement
from RL_utils import Predictor,get_reward_mo,q,search_weights,plt_loss_rew,estimate_and_update,search_thresh,weigh

vocab=to_vocab()
vocab.sort()
vocab_size=len(vocab)

ou2='output/checkpoint_biggest_rnn'
ou1='output/stack_ligand.pkl'
use_cuda = torch.cuda.is_available()
prior=model(vocab_size,use_cuda)
prior.load_model(ou1)
agent=model(vocab_size,use_cuda)
agent.load_model(ou2)

my_Predictor=Predictor(path='datasets/ligand_226.csv',model_path="Exp8_Temperature_" +'\\'+'ECFP4'+'\\'+'ECFP4_phy_True_biao_True'+'_'+'model_')

def main(alg,agent,prior,my_Predictor,epsilon,weight,out,out1,step=None,thresh_set=None,explore=True):
   
    memory_smile=[]
    RL_max = Reinforcement(agent,prior,my_Predictor, get_reward_mo,memory_smile,use_cuda)
    rewards_aver =[]
    rl_losses_aver =[]
    rew_lop=[]
    rew_qed=[]
    rew_sas=[]
    rew_pic=[]
    rew_div=[]
    weight1=[]
    epsilon1=[]
    use_all=np.array([False,False,True,False])
    
    path1 = os.getcwd()
    n_ter=150
    for epoch in range(0,n_ter+1,1):
        if alg=='reward_sum':
            for j in trange(15, desc='Policy gradient...'):
                cur_reward, cur_loss,prop_reward = RL_max.policy_gradient(epsilon,weight)
                if explore:
                    if len(rewards_aver) >2:
                        epsilon = search_thresh(rewards_aver[-3:],thresh_set)
                
                    epsilon1.append(epsilon)  
                
           
                rewards_aver.append(q(rewards_aver, cur_reward)) 
                rl_losses_aver.append(q(rl_losses_aver, cur_loss))
                rew_lop.append(q(rew_lop, prop_reward[0]-0.3))
                rew_qed.append(q(rew_qed, prop_reward[1]))
                rew_sas.append(q(rew_sas, prop_reward[2]-0.1))
                rew_pic.append(q(rew_pic, prop_reward[3]))
            weight= search_weights (rew_lop[-10:],rew_qed[-10:],rew_sas[-10:],rew_pic[-10:],weight,step)
            weight1.append(weight)
        else:
            
            for j in trange(15, desc='Policy gradient...'):
                if alg=='weight_sum':
                    weight=[0.25,0.25,0.25,0.25]
                
                cur_reward, cur_loss,prop_reward = RL_max.policy_gradient(epsilon,weight)
                if explore:
                    if len(rewards_aver) >2:
                        epsilon = search_thresh(rewards_aver[-3:],thresh_set)                
                    epsilon1.append(epsilon) 

                rewards_aver.append(q(rewards_aver, cur_reward)) 
                rl_losses_aver.append(q(rl_losses_aver, cur_loss))
                rew_lop.append(q(rew_lop, prop_reward[0]))
                rew_qed.append(q(rew_qed, prop_reward[1]))
                rew_sas.append(q(rew_sas, prop_reward[2]))
                rew_pic.append(q(rew_pic, prop_reward[3]))
            if alg =='alter_sum':
                
                    
                if epoch % 10 ==0:               
                    use_all=np.roll(use_all,1)
                    a,b,c,d=use_all
                    weight=weigh(a,b,c,d)

        plt_loss_rew(rewards_aver,rl_losses_aver,rew_lop,rew_qed,rew_sas,rew_pic)       
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
    weight=[0,0,0,1]
    
    algs=['reward_sum','alter_sum','weight_sum']
    explores=[True,False]
    for alg in algs:
        for explore in explores:
            if explore:
                thresh_set=1#2,3,
                if alg == 'reward_sum':
                    step=0.005 #0.01
                    out=alg+str(thresh_set)+'_'+str(step)+'.csv'
                    out1=alg+str(thresh_set)+'_'+str(step)
                    main(alg,agent,prior,my_Predictor,epsilon,weight,out,out1,step,thresh_set,explore)
                else:
                    step=None
                    out=alg+str(thresh_set)+'.csv'
                    out1=alg+str(thresh_set)
                    main(alg,agent,prior,my_Predictor,epsilon,weight,out,out1,step,thresh_set,explore)
                    
            else:
                thresh_set=None
                if alg == 'reward_sum':
                    step=0.005 #0.01
                    out=alg+str(epsilon)+'_'+str(step)+'.csv'
                    out1=alg+str(epsilon)+'_'+str(step)
                    main(alg,agent,prior,my_Predictor,epsilon,weight,out,out1,step,thresh_set,explore)
                else:
                    step=None
                    out=alg+str(thresh_set)+'.csv'
                    out1=alg+str(thresh_set)
                    main(alg,agent,prior,my_Predictor,epsilon,weight,out,out1,step,thresh_set,explore)
                
