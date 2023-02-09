import re
import os
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from keras import backend
from sklearn.svm import SVR
from moses.metrics import SA
from rdkit.Chem.QED import qed
from keras import backend as K
from rdkit import Chem,DataStructs
from matplotlib import pyplot as plt
from rdkit.Chem import Descriptors as desc
from rdkit.Chem import AllChem,Lipinski,Crippen
from rdkit.Chem.GraphDescriptors import BertzCT
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold, StratifiedKFold
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint

def data_load(file,chembl_id):
    pair = ['Target_ChEMBL_ID', 'Smiles', 'pChEMBL_Value', 'Comment','Standard_Type', 'Standard_Relation', 'Data_Validity_Comment']
    df = pd.read_table(file,sep=',').dropna(subset=pair[1:2])
    df = df[df[pair[0]] == chembl_id]
    df = df[pair].set_index(pair[1])

    ligand = df[pair[2]].groupby(pair[1]).mean().dropna()

    df = ligand.sample(int(len(ligand)),replace=True)
    return df.index,df.values
def ECFP6_encode(smiles):    
    bit_len=2048
    radius=3
    fps = np.zeros((len(smiles), bit_len))
    phy1=[]
    for i, smile in enumerate(smiles):
        mol = Chem.MolFromSmiles(smile)
        arr = np.zeros((1,))
        phy1.append(mol)
        try:    
            mol = MurckoScaffold.GetScaffoldForMol(mol)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=bit_len)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fps[i, :] = arr
        except:       
            fps[i, :] = [0] * bit_len
    phy=calc_physchem(phy1)
    fp = np.concatenate([fps, phy], axis=1)
    smiles_ECFP6=pd.DataFrame(fp,index=smiles)
    return smiles_ECFP6 
def get_ecfc(smiles_list, radius=3, nBits=2048, useCounts=True):
    """
    Calculates the ECFP fingerprint for given SMILES list
    
    :param smiles_list: List of SMILES
    :type smiles_list: list
    :param radius: The ECPF fingerprints radius.
    :type radius: int
    :param nBits: The number of bits of the fingerprint vector.
    :type nBits: int
    :param useCounts: Use count vector or bit vector.
    :type useCounts: bool
    :returns: The calculated ECPF fingerprints for the given SMILES
    :rtype: Dataframe
    """     
    
    ecfp_fingerprints=[]
    erroneous_smiles=[]
    phy1=[]
    for smiles in smiles_list:
        mol=Chem.MolFromSmiles(smiles)
        phy1.append(mol)
        if mol is None:
            ecfp_fingerprints.append([None]*nBits)
            erroneous_smiles.append(smiles)
        else:
            mol=Chem.AddHs(mol)
            if useCounts:
                ecfp_fingerprints.append(list(AllChem.GetHashedMorganFingerprint(mol, radius, nBits)))  
            else:    
                ecfp_fingerprints.append(list(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits).ToBitString()))  
    
    # Create dataframe of fingerprints
    phy=calc_physchem(phy1) 
           
    f = np.concatenate([ecfp_fingerprints, phy], axis=1)
    df_ecfp_fingerprints = pd.DataFrame(data = f, index = smiles_list)
    # Remove erroneous data
    print(1)
    if len(erroneous_smiles)>0:
        #print("The following erroneous SMILES have been found in the data:\n{}.\nThe erroneous SMILES will be removed from the data.".format('\n'.join(map(str, erroneous_smiles))))           
        df_ecfp_fingerprints = df_ecfp_fingerprints.dropna(how='any')    
    
    return df_ecfp_fingerprints
def ECFP4_encode(smiles):    
    bit_len=2048
    fps = np.zeros((len(smiles), bit_len))
    phy1=[]
    for i, smile in enumerate(smiles):
        mol = Chem.MolFromSmiles(smile)

        phy1.append(mol)
        try:

            fp = Chem.RDKFingerprint(mol, maxPath=4, fpSize=bit_len)           
            DataStructs.ConvertToNumpyArray(fp, fps[i, :])
            

        except:pass
   
    phy=calc_physchem(phy1)        
    f = np.concatenate([fps, phy], axis=1)
    smiles_ECFP4=pd.DataFrame(f,index=smiles)        
    return smiles_ECFP4
def calc_physchem(mols):
    prop_list = ['MW', 'logP', 'HBA', 'HBD', 'Rotable', 'Amide',
                 'Bridge', 'Hetero', 'Heavy', 'Spiro', 'FCSP3', 'Ring',
                 'Aliphatic', 'Aromatic', 'Saturated', 'HeteroR', 'TPSA', 'Valence', 'MR']
    fps = np.zeros((len(mols), 19))
    props = Property()
    for i, prop in enumerate(prop_list):
        props.prop = prop
        fps[:, i] = props(mols)
    return fps
class Property:
    def __init__(self, prop='MW'):
        self.prop = prop
        self.prop_dict = {'MW': desc.MolWt,
                          'logP': Crippen.MolLogP,
                          'HBA': AllChem.CalcNumLipinskiHBA,
                          'HBD': AllChem.CalcNumLipinskiHBD,
                          'Rotable': AllChem.CalcNumRotatableBonds,
                          'Amide': AllChem.CalcNumAmideBonds,
                          'Bridge': AllChem.CalcNumBridgeheadAtoms,
                          'Hetero': AllChem.CalcNumHeteroatoms,
                          'Heavy': Lipinski.HeavyAtomCount,
                          'Spiro': AllChem.CalcNumSpiroAtoms,
                          'FCSP3': AllChem.CalcFractionCSP3,
                          'Ring': Lipinski.RingCount,
                          'Aliphatic': AllChem.CalcNumAliphaticRings,
                          'Aromatic': AllChem.CalcNumAromaticRings,
                          'Saturated': AllChem.CalcNumSaturatedRings,
                          'HeteroR': AllChem.CalcNumHeterocycles,
                          'TPSA': AllChem.CalcTPSA,
                          'Valence': desc.NumValenceElectrons,
                          'MR': Crippen.MolMR,
                          'QED': qed,
                          'SA': SA,
                          'Bertz': BertzCT}
    def __call__(self, mols):
        scores = np.zeros(len(mols))
        for i, mol in enumerate(mols):
            try:
                scores[i] = self.prop_dict[self.prop](mol)
            except:
                continue
        return scores 

class tokens_table(object):
    
    def __init__(self):
        
#        tokens = ['<', '>', '#', '%', ')', '(', '+', '-', '/', '.', '1', '0', '3', '2', '5', '4', '7',
#          '6', '9', '8', '=', 'A', '@', 'C', 'B', 'F', 'I', 'H', 'O', 'N', 'P', 'S', '[', ']',
#          '\\', 'c', 'e', 'i', 'l', 'o', 'n', 'p', 's', 'r', '\n', ' ']
#all_smiles的tok
       #tokens =['#', '%', '(', ')', '+', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', 'A', 'B', 'C', 'F', 'H', 'I', 'L', 'N', 'O', 'P', 'R', 'S', 'T', '[', ']', 'b', 'c', 'e', 'i', 'n', 'o', 'p', 's', 't',' ']
        tokens=['G', '$', '#', '%', ')', '(', '+', '-', '/', '.', '1', '0', '3', '2', '5', '4', '7',
                  '6', '9', '8', '=', 'A', '@', 'C', 'B', 'F', 'I', 'H', 'O', 'N', 'P', 'S', '[', ']',
                  '\\', 'c', 'e', 'i', 'l', 'o', 'n', 'p', 's', 'r', '\n']
        self.table = tokens
        self.table_len = len(self.table)

def SMILES_encode(smiles,max_len=113,phy1=True): 
    
    canons = []
    token = []
    words2=tokens_table().table
    ph=[]
    for i, smile in enumerate(smiles):
        mol = Chem.MolFromSmiles(smile)
       # arr = np.zeros((1,))
       # mol = smiles[i]
        ph.append(mol)
    Dict = dict((c, i) for i, c in enumerate(words2))
    for smile in tqdm(smiles):        
    #    smile='COc1ccc2oc(=O)cc(OCc3cn([C@@H]4c5cc6c(cc5[C@@H](c5cc(OC)c(O)c(OC)c5)[C@H]5C(=O)OC[C@H]45)OCO6)nn3)c2c1'
        regex = '(\[[^\[\]]{1,6}\])'
        smile1 = smile
        tokens=[]
        for word in re.split(regex, smile1): 
    #    for word in smile:
            if word == '' or word is None: continue
            
            '''if word.startswith('['):
                tokens.append(word)
            else:'''
            for i, char in enumerate(word):
                tokens.append(char)
        if len(tokens)<=max_len:
            dif = max_len - len(tokens)
            [tokens.append(' ') for _ in range(dif)]                     
        token.append(tokens)
    newSmiles =  np.zeros((len(token), len(token[0])))
    for i in range(0,len(token)):
        for j in range(0,len(token[i])):
            
            newSmiles[i,j] = Dict[token[i][j]]
    phy=calc_physchem(ph)
    new_tok_X1 = np.concatenate([newSmiles, phy], axis=1)
    smiles_tok=pd.DataFrame(newSmiles,index=smiles)
    smiles_tok1=pd.DataFrame(new_tok_X1,index=smiles)
    if phy1:
        return smiles_tok1 
    return smiles_tok  

def Train_Test(smiles,label):   
    np.random.seed(64)
    msk = np.random.rand(len(smiles)) < 0.9
    labels = pd.DataFrame({'PCHMEL': label[:]})        
    labels_df_train_val = labels[msk]
    labels_train=labels_df_train_val.to_numpy()
    labels_train1,_=normalize(labels_train) 
    labels_df_test = labels[~msk]
    labels_test=labels_df_test.to_numpy()
    labels_test1,_=normalize(labels_test)
    train = smiles[msk]
    test = smiles[~msk]
    test=np.array(test)
    return labels_train,labels_train1,labels_test,labels_test1,train,test

def normalize(data_y):
    data_aux = np.zeros(2)

    q1_train = np.percentile(data_y, 5)
    q2_train = np.percentile(data_y, 90)
    
   # q3_train = np.percentile(data_y, 90)
    data_y=(data_y - q1_train) / (q2_train - q1_train)

    data_aux[1] = q1_train
    data_aux[0] = q2_train
    return data_y,data_aux
def denormalization1(predictions,labels):   
    
    for l in range(len(predictions)):
        
        q1 = np.percentile(labels,5)
        q3 = np.percentile(labels,95)
       
        for c in range(len(predictions[0])):
            predictions[l,c] = (q3 - q1) * predictions[l,c] + q1
#            predictions[l,c] = predictions[l,c] * sd_train + m_train
          
    return predictions

def cross_validation_split(x, y, n_folds=5, split='random', folds=None):
    assert(len(x) == len(y))
    x = np.array(x)
    y = np.array(y)
    
    if split == 'random':
        cv_split = KFold(n_splits=n_folds, shuffle=True)
        folds = list(cv_split.split(x, y))
    elif split == 'stratified':
        cv_split = StratifiedKFold(n_splits=n_folds, shuffle=True)
        folds = list(cv_split.split(x, y))
    elif split == 'fixed' and folds is None:
        raise TypeError(
            'Invalid type for argument \'folds\': found None, but must be list')
    cross_val_data = []
    cross_val_labels = []
    if len(folds) == n_folds:
        for fold in folds:
            cross_val_data.append(x[fold[1]])
            cross_val_labels.append(y[fold[1]])
    elif len(folds) == len(x) and np.max(folds) == n_folds:
        for f in range(n_folds):
            left = np.where(folds == f)[0].min()
            right = np.where(folds == f)[0].max()
            cross_val_data.append(x[left:right + 1])
            cross_val_labels.append(y[left:right + 1])

    return cross_val_data, cross_val_labels

def grid_search_param_machine(X,Y,model_type,use_cuda):
    if model_type=='RF':        
        model=RandomForestRegressor()
        gs = GridSearchCV(model, {'n_estimators': [1200,1500,1700],'max_features': ["auto", "sqrt", "log2"]})
        gs.fit(X,Y)
        params = gs.best_params_
    elif model_type=='SVR': 
        model = SVR()             
        gs = GridSearchCV(model, {'kernel': ['rbf','linear','poly'], 'C': 2.0 ** np.array([-3, 13]), 'gamma': 2.0 ** np.array([-13, 3])}, n_jobs=5)
        gs.fit(X,Y)
        params = gs.best_params_
    elif model_type=='KNN':       
        model = KNeighborsRegressor()       
        gs = GridSearchCV(model, {'n_neighbors': [1,2,3,4], 'metric': ['euclidean', 'manhattan', 'chebyshev']})
        gs.fit(X,Y)
        params = gs.best_params_
    elif model_type=='PLS':        
        model=PLSRegression()
        gs = GridSearchCV(model, {'n_components': [1,2,3],'max_iter':[400,500,600], 'tol':[1e-07,1e-06,1e-05]})
        gs.fit(X,Y)
        params = gs.best_params_    
    return params

def Model(X,Y,x_val,y_val,size,model_type,feature,i):
    
    path=os.getcwd() 
    
    fold="Exp8_Temperature_" +'\\'+model_type+'\\'
    file=path+"\\"+fold
    if os.path.exists(file):
        pass
    else:
        os.makedirs(file)
    filename=file+feature+'_'+'model_240'+str(i)
    
   
    model = Sequential()
    model.add(Input(shape=(size,)))
    model.add(Dense(8000, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(4000, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(2000, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1, activation='linear'))
    opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.99, amsgrad=False)
            	
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, restore_best_weights=True)
    mc = ModelCheckpoint(filename, monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    model.compile(loss="mean_squared_error", optimizer = opt, metrics=[r_square,rmse,ccc])
#        lrateh_size=self.config.batch_size,validation_data=(X_test, y_test),callbacks=[lrate])
    result = model.fit(X, Y,epochs=100,batch_size=16,validation_data=(x_val, y_val),callbacks=[es,mc])
    model.summary()    
    plt.plot(result.history['r_square'])
    plt.plot(result.history['val_r_square'])
    plt.title('model R^2')
    plt.ylabel('R^2')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()                   
        # plot training curve for rmse
    plt.plot(result.history['rmse'])
    plt.plot(result.history['val_rmse'])
    plt.title('rmse')
    plt.ylabel('rmse')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
   
#             serialize model to JSON
    model_json = model.to_json()
    with open(str(filename + ".json"), "w") as json_file:
        json_file.write(model_json)
                # serialize weights to HDF5
    model.save_weights(str(filename + ".h5"))
    print("Saved model to disk")

def rmse(y_true, y_pred):
  
    
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

def mse(y_true, y_pred):

    
    return backend.mean(backend.square(y_pred - y_true), axis=-1)

def r_square(y_true, y_pred):

    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return (1 - SS_res/(SS_tot + K.epsilon()))


def ccc(y_true,y_pred):

 
    num = 2*K.sum((y_true-K.mean(y_true))*(y_pred-K.mean(y_pred)))
    den = K.sum(K.square(y_true-K.mean(y_true))) + K.sum(K.square(y_pred-K.mean(y_pred))) + K.int_shape(y_pred)[-1]*K.square(K.mean(y_true)-K.mean(y_pred))
    return num/den

def model_train(X,Y,val_X,val_Y,model_type,param,i,use_cuda,phy=True,biaozhun=True,feature=None):
    path=os.getcwd()    
    if model_type=='RF':       
        model = RandomForestRegressor(n_estimators=param['n_estimators'], max_features = param['max_features'], random_state = 0, n_jobs=-1)               
        pred = model.fit(X,Y).predict(val_X)
    elif model_type=='SVR':        
        model = SVR(kernel=param['kernel'], C=param['C'], gamma=param['gamma'])             
        pred = model.fit(X,Y).predict(val_X)
    elif model_type=='KNN':
        model = KNeighborsRegressor(n_neighbors=param['n_neighbors'], metric = param['metric'], n_jobs= 1)              
        pred = model.fit(X,Y).predict(val_X)
    elif model_type=='PLS':
        model = PLSRegression(n_components=param['n_components'],max_iter=param['max_iter'],tol=param['tol'])
        pred=model.fit(X,Y).predict(val_X)  
    r2 =r_square(val_Y, pred)
    mse1=mse(val_Y, pred)   
    # Save the model as a pickle in a file
    print('train'+str(mse1)+'_'+str(r2))
    fold="Exp8_Temperature_" +'\\'+model_type+'\\'
    file=path +"\\"+ fold
    if os.path.exists(file):
        pass
    else:
        os.makedirs(file)
    if phy:
        phy1='phy_True'
    else:phy1='phy_False'
    if biaozhun:
        biaozhun1='biao_True'
    else:biaozhun1='biao_False'    
    #filename=feature+'_'+'_'+'model_'+ n +'_'+"9.pkl" 
    filename=feature+'_'+phy1+'_'+biaozhun1+'_'+'model_226'+str(i) +'.pkl' 
    filepath=file+F"{filename}"
    joblib.dump(model, filepath)

def regression_plot(y_true,y_pred):   
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred)
    ax.plot([np.min(y_true), np.max(y_true)], [np.min(y_true), np.max(y_true)], 'k--', lw=4)
    ax.set_xlabel('True')
    ax.set_ylabel('Predicted')
    plt.show()

