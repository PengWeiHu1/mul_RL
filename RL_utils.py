import torch
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from rdkit import Chem
import matplotlib.pyplot as plt
from moses.metrics import QED, SA, logP
from pre_util import denormalization1,ECFP4_encode
from utils import sampleing,validity,remove_duplicates,char_tensor,diversity
class Predictor(object):
    def __init__(self,path,model_path):       
        super(Predictor, self).__init__()
        self.labels= pd.read_table(path,sep=',').PIC.dropna()       
        self.ligand = np.asanyarray(self.labels).reshape(-1)  
        self.label=self.ligand.tolist()
        self.label1=[]
        for i in range(len(self.label)):
            self.label1.append(float(self.label[i]))                  
        self.model = []         
        for i in range(5):            
            self.model.append(joblib.load(model_path+str(i)+'.pkl'))                            
    def predict(self, smiles):
        prediction1=[]
        smiles_encode=ECFP4_encode(smiles)
        smiles_encode=np.array(smiles_encode)
        for i in range(5):
            y_pred = self.model[i].predict(smiles_encode)
            prediction1.append(y_pred)
        prediction = np.array(prediction1).reshape(5, -1)
        prediction = denormalization1(prediction,self.label1)
                    
        prediction = np.mean(prediction, axis = 0)
        return prediction

def get_reward_mo(smiles,my_Predictor,memory_smiles):
    
    rew=[]
    #log p
    mol = Chem.MolFromSmiles(smiles[0])
    lop = logP(mol)
    if lop > 1 and lop < 4:
        reward_logP = 1
    else:
        reward_logP = 0
    rew.append(reward_logP)
    #qed
    Q = QED(mol)
    reward_qed = np.exp(Q/3)
    rew.append(reward_qed)
    #sa
    sas_list = SA(mol)
    reward_sa = np.exp(-sas_list/4 + 2)   
    rew.append(reward_sa)
    #pic 
    
    
    prop = my_Predictor.predict(smiles)
    reward_pic=np.exp(prop[0]/3-1)
    rew.append(reward_pic)
    #div
   
    if len(memory_smiles) >= 30:
        diversity1 = diversity(smiles,memory_smiles)
        
    if diversity1 < 0.8:
        rew_div = 0.01
        print("\Alert: Similar compounds")
    
    else :
    
        rew_div=1 
    
    rew.append(rew_div)   
    return rew

def q(previous_values, new_value, ma_window_size=10):
    value_ma = np.sum(previous_values[-(ma_window_size-1):]) + new_value
    value_ma = value_ma/(len(previous_values[-(ma_window_size-1):]) + 1)
    return value_ma

def search_weights(rew_lop,rew_qed,rew_sas,rew_pic,inti_weights,step):
    rew_logp=np.mean(rew_lop)
    rew_qed=np.mean(rew_qed)
    rew_sas=np.mean(rew_sas)
    rew_pic=np.mean(rew_pic)
    t=[[rew_logp,'a',inti_weights[0]],[rew_qed,'b',inti_weights[1]],[rew_sas,'c',inti_weights[2]],[rew_pic,'d',inti_weights[3]]]

    t=sorted(t)
    #改变最小奖励的权重

    if t[0][1]=='a':
        logp_weight= inti_weights[0]+step
        inti_weights[0]=np.clip(logp_weight,0,1)
    elif t[0][1]=='b': 
        qed_weight= inti_weights[1]+step
        inti_weights[1]=np.clip(qed_weight,0,1)
    elif t[0][1]=='c':
        sas_weight= inti_weights[2]+step
        inti_weights[2]=np.clip(sas_weight,0,1)
    elif t[0][1]=='d':
        pic_weight= inti_weights[3]+step
        inti_weights[3]=np.clip(pic_weight,0,1)
     #改变最大奖励的权重   
    if t[3][1]=='a':
        logp_weight= inti_weights[0]-step
        inti_weights[0]=np.clip(logp_weight,0,1)
    elif t[3][1]=='b': 
        qed_weight= inti_weights[1]-step
        inti_weights[1]=np.clip(qed_weight,0,1)
    elif t[3][1]=='c':
        sas_weight= inti_weights[2]-step
        inti_weights[2]=np.clip(sas_weight,0,1)
    elif t[3][1]=='d':
        pic_weight= inti_weights[3]-step
        inti_weights[3]=np.clip(pic_weight,0,1)
        
    if t[3][2]==0  :
        if t[2][1]=='a':
            logp_weight= inti_weights[0]-step
            inti_weights[0]=np.clip(logp_weight,0,1)
        elif t[2][1]=='b': 
            qed_weight= inti_weights[1]-step
            inti_weights[1]=np.clip(qed_weight,0,1)
        elif t[2][1]=='c':
            sas_weight= inti_weights[2]-step
            inti_weights[2]=np.clip(sas_weight,0,1)
        elif t[2][1]=='d':
            pic_weight= inti_weights[3]-step
            inti_weights[3]=np.clip(pic_weight,0,1)
    if t[2][2] ==0 and t[3][2]==0 :
        if t[1][1]=='a':
            logp_weight= inti_weights[0]-step
            inti_weights[0]=np.clip(logp_weight,0,1)
        elif t[1][1]=='b': 
            qed_weight= inti_weights[1]-step
            inti_weights[1]=np.clip(qed_weight,0,1)
        elif t[1][1]=='c':
            sas_weight= inti_weights[2]-step
            inti_weights[2]=np.clip(sas_weight,0,1)
        elif t[1][1]=='d':
            pic_weight= inti_weights[3]-step
            inti_weights[3]=np.clip(pic_weight,0,1)
    inti_weights=[inti_weights[0],inti_weights[1],inti_weights[2],inti_weights[3]]
  
    return inti_weights

def plt_loss_rew(rewards_aver,rl_losses_aver,rew_lop,rew_qed,rew_sas,rew_pic):
    plt.plot(rew_lop)
    plt.xlabel('Training iteration')
    plt.ylabel('average rew_lop')
    plt.show()
    
    plt.plot(rew_qed)
    plt.xlabel('Training iteration')
    plt.ylabel('Average rew_qed')
    plt.show() 

    plt.plot(rew_sas)
    plt.xlabel('Training iteration')
    plt.ylabel('average rew_sas')
    plt.show()
    
    plt.plot(rew_pic)
    plt.xlabel('Training iteration')
    plt.ylabel('Average rew_pic')
    plt.show() 
    
    plt.plot(rl_losses_aver)
    plt.xlabel('Training iteration')
    plt.ylabel('Loss')
    plt.show()
    
    plt.plot(rewards_aver)
    plt.xlabel('Training iteration')
    plt.ylabel('Average reward')
    plt.show()       

def plot_hist(prediction, n_to_generate,prop):

    prediction = np.array(prediction)
    x_label = ''
    plot_title = '' 

    if prop == "pic":
        print("Max of pIC50: ", np.max(prediction))
        print("Mean of pIC50: ", np.mean(prediction))
        print("Min of pIC50: ", np.min(prediction))
        percentage_in_pic = np.sum(prediction >=6.5 )/len(prediction)
        x_label = "Predicted pIC50"
        plot_title = "Distribution of predicted pIC50 for generated molecules"
    elif prop == "sas":
        print("Max SA score: ", np.max(prediction))
        print("Mean SA score: ", np.mean(prediction))
        print("Min SA score: ", np.min(prediction))
        x_label = "Calculated SA score"
        plot_title = "Distribution of SA score for generated molecules"
    elif prop == "qed":
        print("Max QED: ", np.max(prediction))
        print("Mean QED: ", np.mean(prediction))
        print("Min QED: ", np.min(prediction))
        x_label = "Calculated QED"
        plot_title = "Distribution of QED for generated molecules"  
        
    elif prop == "logp":
        percentage_in_threshold = np.sum((prediction >= 0.0) & 
                                     (prediction <= 5.0))/len(prediction)
        print("Percentage of predictions within drug-like region:", percentage_in_threshold)
        print("Average of log_P: ", np.mean(prediction))
        print("Median of log_P: ", np.median(prediction))
        plt.axvline(x=0.0)
        plt.axvline(x=5.0)
        x_label = "Predicted LogP"
        plot_title = "Distribution of predicted LogP for generated molecules"
        
#    sns.set(font_scale=1)
    ax = sns.kdeplot(prediction, shade=True,color = 'g')
    ax.set(xlabel=x_label,
           title=plot_title)
    plt.show()
    if prop == "logp":
        return percentage_in_threshold
    if prop == "pic":
        return percentage_in_pic
def lopg_sa_qed(smiles):
    lopg=[]
    
    sa=[]
    for smil in smiles:
        mol = Chem.MolFromSmiles(smil)
        lop = logP(mol)
        
        sas_list = SA(mol)
        lopg.append(lop)
        
        sa.append(sas_list)
    qed=[]
    for smil in smiles:
        try:
            mol = Chem.MolFromSmiles(smil)
            Q = QED(mol)
            qed.append(Q)
        except:
            print('Invalid')
    return lopg,sa,qed
    
def estimate_and_update(generator,my_Predictor,use_cuda,vocab):    
    n_to_generate=500
    generated = []
    pbar = tqdm(range(n_to_generate))
    for i in pbar:
        pbar.set_description("Generating molecules...")
        generated.append(sampleing(generator,use_cuda,vocab)[1:-1])
    generate=[]
    for smil in generated:    
        smile = smil.replace('R', 'Br').replace('L', 'Cl')
        generate.append(smile)                                     
    valid_smiles, val_perc = validity(generate)
    smiles_list,uniq=remove_duplicates(valid_smiles) 
    int_div = diversity(smiles_list)
    pre=my_Predictor.predict(smiles_list)
#    print("Mean value of predictions:", prediction.mean())
    lopg,sa,qed=lopg_sa_qed(smiles_list)

    #    print("Mean value of predictions:", prediction.mean())
    print("Proportion of valid SMILES:", val_perc)
    print("Proportion of uniq SMILES:", uniq)
    print("Proportion of div SMILES:", int_div)
    logp_valid=plot_hist(lopg,n_to_generate,'logp')
    
    plot_hist(qed,n_to_generate,'qed')
    plot_hist(sa,n_to_generate,'sas')
    pic_valid=plot_hist(pre,n_to_generate,'pic')    
    pr_logp = np.array(lopg)    
    pred_logp=pr_logp.mean()    
    pr_qed = np.array(qed)    
    pred_qed=pr_qed.mean()    
    pr_sa = np.array(sa)    
    pred_sa=pr_sa.mean()   
    pr_pic = np.array(pre)    
    pred_pic=pr_pic.mean()
    return smiles_list,val_perc,uniq,int_div,pred_logp,logp_valid,pred_qed,pred_sa,pred_pic,pic_valid

def search_thresh(reward,thresh_set):
    r_2=reward[0]
    r_1=reward[1]
    r = reward[2]
    q1 = r_2/r_1
    q2 = r_1 / r
    
    if thresh_set == 1:
        thresholds_set=[0,0.001,0.0001]
    elif thresh_set ==2 :
         thresholds_set= [0.0001,0.01,0.001]  
    elif thresh_set ==3 :
         thresholds_set= [0.001,0.05,0.01]
    elif thresh_set == 5:
        thresholds_set =[0.05,0.15,0.1]
    elif thresh_set ==4:
        thresholds_set =[0.01,0.1,0.05]

    if q1 < 1 and q2 < 1:
        threshold = thresholds_set[0]
    elif q1 > 1 and q2 > 1:
        threshold = thresholds_set[1]
    else:
        threshold = thresholds_set[2]
    return threshold 

def weigh(a,b,c,d):
    if a:
        weight=[1,0,0,0]
    elif b:
        weight=[0,1,0,0]
    elif c:
        weight=[0,0,1,0]
    elif d:
        weight=[0,0,0,1]
    return weight

def scare(rewards,i,weights):
    r=[]
    pred_range=[6.01,2.6203,1.39,1.2214]
    w_lop = weights[0]
    w_qed = weights[1]
    w_sas = weights[2]
    w_pic = weights[3]
    
    rew_logp = rewards[0]
    r.append(rew_logp)
    rew_qed = rewards[1]
    rew_sas = rewards[2]
    rew_pic = rewards[3]
    rew_div = rewards[4]
    
    max_pic = pred_range[0]
    min_pic = pred_range[1]
    max_qed = pred_range[2]
    min_qed = pred_range[3]
    max_sas = 5.83
    min_sas = 2.2
    scale_qed = (rew_qed - min_qed)/(max_qed-min_qed)
    scale_qed = np.clip(scale_qed,0,1)
    r.append(scale_qed)
    scale_sas = (rew_sas - min_sas)/(max_sas-min_sas)
    scale_sas=np.clip(scale_sas,0,1)
    r.append(scale_sas)
    
    scale_pic = (rew_pic - min_pic)/(max_pic-min_pic)  
    scale_pic=np.clip(scale_pic,0,1)
    r.append(scale_pic)
    r.append(rew_div)
    rew=(w_pic * scale_pic + w_qed * scale_qed + w_lop* rew_logp + w_sas*scale_sas)*3.1*rew_div
        
    return rew,r


def RL_sample(model,model1,use_cuda,vocab,epsilon=None,sample=None,prime_str = 'G', end_token = '$', predict_len=100):     
    hidden = model.init_hidden()
    if model.has_cell:
        cell = model.init_cell()
        hidden = (hidden, cell)
    if model.has_stack:
        stack = model.init_stack()
    else:
        stack = None
    hidden1 = model1.init_hidden()
    if model1.has_cell:
        cell1 = model1.init_cell()
        hidden1 = (hidden1, cell1)
    if model1.has_stack:
        stack1 = model1.init_stack()
    else:
        stack = None
    prime_input = char_tensor(prime_str,use_cuda,vocab)
   
    new_sample = prime_str
        # Use priming string to "build up" hidden state
    for p in range(len(prime_str)-1):
        
#        print(p)
        _, hidden, stack = model.forward(prime_input[p], hidden, stack)
        _, hidden1, stack1 = model1.forward(prime_input[p], hidden1, stack1)
        
    inp = prime_input[-1]

    for p in range(predict_len):
        
        output, hidden, stack = model1.forward(inp, hidden, stack)                
            
        #probas = torch.softmax(output, dim=1)
        output1, hidden1, stack1 = model.forward(inp, hidden1, stack1) 
            # Sample from the network as a multinomial distribution
        ratio = torch.rand(1, 1).cuda() * epsilon
        probas = torch.softmax(output1,dim=1) * (1 - ratio) + torch.softmax(output,dim=1) * ratio
        top_i = torch.multinomial(probas.view(-1), 1)[0].cuda()#.numpy()

            # Add predicted character to string and use as next input
        predicted_char = vocab[top_i]
        if predicted_char==prime_str:
            new_sample +=end_token
            break
        new_sample += predicted_char
       
        inp =char_tensor(predicted_char,use_cuda,vocab)
        if predicted_char == end_token:
           break
        
    return new_sample

def get_reward_logP(smiles,my_Predictor,memory_smiles):
    
    mol = Chem.MolFromSmiles(smiles[0])
    pred = logP(mol)
    if pred > 1 and pred < 4:
        reward_logP = 11
    else:
        reward_logP = 1

    
    if len(memory_smiles) >= 30:
        diversity1 = diversity(smiles,memory_smiles)
        
    if diversity1 < 0.75:
        rew_div = 0.01
        print("\Alert: Similar compounds")
    else :
    
        rew_div=1 
    
    rewards = reward_logP*rew_div
    return rewards

def get_reward_pIC(smiles, my_Predictor,memory_smiles):

    prop = my_Predictor.predict(smiles)
    rew_pic=np.exp(prop[0]/3-1)
    
    if len(memory_smiles) >= 30:
        diversity1 = diversity(smiles,memory_smiles)
        
    if diversity1 < 0.8:
        rew_div = 0.01
        print("\Alert: Similar compounds")
   

    else :
    
        rew_div=1 
    rewards = rew_div*rew_pic      
    return rewards









