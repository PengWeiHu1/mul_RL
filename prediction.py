import re
import torch
import joblib
import numpy as np
import pandas as pd 
from rdkit import Chem    
from tqdm import tqdm
from pre_util import *
from time import process_time
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import model_from_json
from rdkit.Chem.MolStandardize import rdMolStandardize
from sklearn.preprocessing import MinMaxScaler as Scaler

def main(filename,trgs,model,phy,out):
    
    data_x,data_y=data_load(filename,trgs)
    smiles=[]
    for smile in tqdm(data_x):
        try:
            mol = Chem.MolFromSmiles(smile)
            mol = rdMolStandardize.ChargeParent(mol)        
            if mol is not None:
                mo=Chem.MolToSmiles(mol,isomericSmiles=False)
            else:mo=''
            smiles.append(Chem.CanonSmiles(mo))
        except:
            print(smile)

    words = set()
    canons = []
    canons1 = []
    token = []
    for smile in tqdm(smiles):
    
#    smile='COc1ccc2oc(=O)cc(OCc3cn([C@@H]4c5cc6c(cc5[C@@H](c5cc(OC)c(O)c(OC)c5)[C@H]5C(=O)OC[C@H]45)OCO6)nn3)c2c1'
        regex = '(\[[^\[\]]{1,6}\])'
        smile1 = smile
        tokens=[]#['[G]']
        for word in re.split(regex, smile1): 
#    for word in smile:
            if word == '' or word is None: continue
        
            if word.startswith('['):
                tokens.append(word)
            else:
                for i, char in enumerate(word):
                    tokens.append(char)
    #tokens.append('[A]')
   
        words.update(tokens)       
        canons.append(smile)
        token.append(tokens)
        canons1.append(smile1)

    smiles = []
    labels = []
    smiles_tihuan=[]
    for i in range(len(token)):
        if 10 <len(canons1[i]) <= 100:  
            smiles.append(canons[i])
            smiles_tihuan.append(canons1[i])
            labels.append(data_y[i])

    maxlen=len(max(smiles_tihuan, key=len))

    data_x=smiles
    data_y=labels

    use_cuda = True

    if use_cuda is None:
        use_cuda = torch.cuda.is_available()
#%%
    model=['RF']#,'PLS','SVR']#,'dnn','rnn']KNN RF 'KNN','RF'
    phy=True#False
#from pre_util import ccc,mse,rmse,r_square

    for model_type in model:
        if model_type=='dnn':
            features=['ECFP4']
    
        else:features=['ECFP4','SMILES']
        if phy:
            features=features
        else:features=['SMILES']
    
        for feature in features:
       

            if feature=='SMILES':      
                smiles_encode=SMILES_encode(data_x,maxlen,phy)
            elif feature=='ECFP4':
                smiles_encode=ECFP4_encode(data_x)
        
            
            #smiles_encode=torch.from_numpy(smiles_encode)
            #data_y=torch.from_numpy(data_y)
            labels_train,labels_train1,labels_test,labels_test1,train,test=Train_Test(smiles_encode,data_y)
            for biaozhun in [False,True]:
                if biaozhun==True:
                    scaler = Scaler(); scaler.fit(train)
                    test = scaler.transform(test)
                    train = scaler.transform(train)
#if predict_model=='machine':
                start_train_t = process_time()
                if model_type == 'RF' or model_type=='SVR' or model_type=='KNN' or model_type=='PLS':
                    params=grid_search_param_machine(train, labels_train,model_type,use_cuda)
                    print(1)
                cross_val_data, cross_val_labels = cross_validation_split(x=train, y=labels_train,split='random',n_folds=5)            
                for i in range(5):
                    #shuzi=['O','I','II','III','IV']
                    k=int(i)
                    train_x = np.concatenate(cross_val_data[:i] +cross_val_data[(i + 1):])
                    val_x = cross_val_data[i]
                    train_y = np.concatenate(cross_val_labels[:i] +cross_val_labels[(i + 1):])
                    val_y = cross_val_labels[i]
            #模型对验证集评估与保存与标准化
                    train_y1,data_norcan=normalize(train_y)
                    val_y1,_=normalize(val_y)
                    if model_type=='dnn': 
                        size=2067
                        Model(train_x,train_y1,val_x,val_y1,size,model_type,feature,k)
                    else:
                        model_train(train_x,train_y1,val_x,val_y1,model_type,params,k,use_cuda,phy,biaozhun,feature)  
                end_train_t = process_time()
                train_time = end_train_t - start_train_t
                metrics1=[]
                prediction1=[] 
                file="Exp8_Temperature_" +'\\'+model_type+'\\'
                for i in range(5):
                    if model_type=='dnn':
                        json_file = open(file+feature+'_'+ "model_226"+str(i)+".json", 'r')
                        loaded_model_json = json_file.read()
                        json_file.close()
                        loaded_model = model_from_json(loaded_model_json)
                # load weights into new model
                        loaded_model.load_weights(file+feature+'_'+ "model_226"+str(i)+".h5")
                        opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.99, amsgrad=False)
                        loaded_model.compile(loss="mean_squared_error", optimizer = opt, metrics=[mse,r_square,rmse,ccc])
                        metrics1.append(loaded_model.evaluate(test,labels_test1))
                        prediction1.append(loaded_model.predict(test))
                    else:
                        if phy:
                            phy1='phy_True'
                        else:phy1='phy_False'
                        if biaozhun:
                            biaozhun1='biao_True'
                        else:biaozhun1='biao_False'             
                        loaded_model = joblib.load(file+feature+'_'+phy1+'_'+biaozhun1+'_'+'model'+str(i)+'.pkl') 
                        y_pred = loaded_model.predict(test)
                        r2_sc = r_square(labels_test1, y_pred)
                        ms_error = mse(labels_test1, y_pred)
            #                rms_error = rmse(label,y_pred)
            #                ccc_value = ccc(label,y_pred)
                
                        print("mse_Q2: ",[ms_error,r2_sc])
                        metrics1.append([ms_error,r2_sc])
                        prediction1.append(y_pred)
                print(train_time)
                prediction = np.array(prediction1).reshape(5, -1)
                prediction2 = np.mean(prediction, axis = 0)
                regression_plot(labels_test1,prediction2) 
                if model_type=='dnn':
                    metrics1 = np.array(metrics1).reshape(5, -1)
                    metrics1 = metrics1[:,1:5]
                    mean_metrics = np.mean(metrics1, axis = 0)
                    values=[model_type,feature,mean_metrics,train_time]
                else:
                    mean_metrics = np.mean(metrics1, axis = 0)
                    values=[model_type,feature,phy,biaozhun,params,mean_metrics,train_time]
                file=[i.rstrip().split(',') for i in open(out).readlines()]
                file.append(values)
                file=pd.DataFrame(file)
                file.to_csv(out,header=None,index=None)

if __name__ == '__main__':
    filename='datasets/ligand.csv'
    trgs='CHEMBL226'
    model=['RF']#,'PLS','SVR']#,'dnn','rnn']KNN RF 'KNN','RF'
    phy=True#False
    out='grid_results_ji_machine.csv'
    main(filename,trgs,model,phy,out)




















