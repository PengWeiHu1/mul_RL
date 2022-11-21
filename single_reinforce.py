import torch
import random
import numpy as np
from rdkit import Chem
from RL_utils import RL_sample
import torch.nn.functional as F
from utils import to_vocab,char_tensor

class Reinforcement(object):
    def __init__(self, generator,ub_generator, predictor, get_reward,memorysmiles,use_cuda):

        super(Reinforcement, self).__init__()
        self.generator = generator
       
        self.ub_generator = ub_generator
        self.predictor = predictor
        self.get_reward = get_reward
        self.use_cuda=use_cuda
        self.memorysmiles=memorysmiles
        
        vocab=to_vocab()
        self.vocab=vocab

    def policy_gradient(self,epsilon=None,
                        std_smiles=False, grad_clipping=None, **kwargs):
       
        rl_loss = 0
        self.generator.optimizer.zero_grad()
        total_reward = 0
        
       
            
        for _ in range(10):
            # Sampling new trajectory
            reward = 0
            #trajectory = '<>'
            while reward == 0:
                while len(self.memorysmiles) < 31:
                    trajectory = RL_sample(self.generator,self.ub_generator,self.use_cuda,self.vocab,epsilon)
                    smile = trajectory
                    sm1=smile[1:-1]                                   
                    if len(sm1)<100 and len(set(sm1))>4:                     
                        if Chem.MolFromSmiles(sm1) != None and sm1 !='':                             
                            if sm1 not in self.memorysmiles:                              
                                self.memorysmiles.append(sm1) 
                print(len(self.memorysmiles))
                        
                sm1=self.memorysmiles[-1]
               
        
                try:                                                                     
                    if Chem.MolFromSmiles(sm1) != None and sm1 !='': 
                        self.memorysmiles.remove(self.memorysmiles[0])
                        reward = self.get_reward([sm1],self.predictor,self.memorysmiles)  
                        print(reward)
                    else:
                        reward=0
                        print('invalid1')
                except:
                    reward=0
                    self.memorysmiles.remove(sm1)
                    print("invalid2")
       
            sm2 = sm1 
            trajectory='G'+sm2+'$'    
            # Converting string of characters into tensor
            trajectory_input =char_tensor(trajectory,self.use_cuda,self.vocab)
            discounted_reward = reward
            total_reward += reward

            # Initializing the generator's hidden state
            hidden = self.generator.init_hidden()
            if self.generator.has_cell:
                cell = self.generator.init_cell()
                hidden = (hidden, cell)
            if self.generator.has_stack:
                stack = self.generator.init_stack()
            else:
                stack = None

            # "Following" the trajectory and accumulating the loss
            for p in range(len(trajectory)-1):
                output, hidden, stack = self.generator(trajectory_input[p], 
                                                       hidden, 
                                                       stack)
                log_probs = F.log_softmax(output, dim=1)
                top_i = trajectory_input[p+1]
                rl_loss -= (log_probs[0, top_i]*discounted_reward)
                discounted_reward = discounted_reward * 0.97

        # Doing backward pass and parameters update
        rl_loss = rl_loss / 10
        total_reward = total_reward / 10
        rl_loss.backward()
        if grad_clipping is not None:
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 
                                           grad_clipping)

        self.generator.optimizer.step()
        
        return total_reward, rl_loss.item()



