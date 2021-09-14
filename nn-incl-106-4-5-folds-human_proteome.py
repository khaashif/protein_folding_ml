#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 20:19:07 2021

@author: khaashif
"""


# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 20:01:32 2021

@author: fayya
"""
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import DataStructs
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm
from sklearn.model_selection import StratifiedKFold
import itertools

from Bio import SeqIO
from Bio.SeqIO import FastaIO
from itertools import product
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
import math


import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics.pairwise import rbf_kernel as kernel  # sigmoid_kernel,rbf_kernel,linear_kernel
from sklearn.model_selection import StratifiedKFold, KFold
import random
from keras import Sequential
from keras.layers import Dense
import keras
from keras.layers.core import Dropout

#Function to convert compounds into features
def getFP(s, r=3, nBits=1024):
    compound = Chem.MolFromSmiles(s.strip())
    fp = AllChem.GetMorganFingerprintAsBitVect(compound, r, nBits=nBits)
    # fp = pat.GetAvalonCountFP(compound,nBits=nBits)
    m = np.zeros((0,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, m)
    return m

#Function to convert proteins into features
def twomerFromSeq(s):
    k = 2
    groups = {'A': '1', 'V': '1', 'G': '1', 'I': '2', 'L': '2', 'F': '2', 'P': '2', 'Y': '3',
              'M': '3', 'T': '3', 'S': '3', 'H': '4', 'N': '4', 'Q': '4', 'W': '4',
              'R': '5', 'K': '5', 'D': '6', 'E': '6', 'C': '7'}
    crossproduct = [''.join(i) for i in product("1234567", repeat=k)]
    for i in range(0, len(crossproduct)): crossproduct[i] = int(crossproduct[i])
    ind = []
    for i in range(0, len(crossproduct)): ind.append(i)
    combinations = dict(zip(crossproduct, ind))

    V = np.zeros(int((math.pow(7, k))))  # defines a vector of 343 length with zero entries
    try:
        for j in range(0, len(s) - k + 1):
            kmer = s[j:j + k]
            c = ''
            for l in range(0, k):
                c += groups[kmer[l]]
                V[combinations[int(c)]] += 1
    except:
        count = {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0}
        for q in range(0, len(s)):
            if s[q] == 'A' or s[q] == 'V' or s[q] == 'G':
                count['1'] += 1
            if s[q] == 'I' or s[q] == 'L' or s[q] == 'F' or s[q] == 'P':
                count['2'] += 1
            if s[q] == 'Y' or s[q] == 'M' or s[q] == 'T' or s[q] == 'S':
                count['3'] += 1
            if s[q] == 'H' or s[q] == 'N' or s[q] == 'Q' or s[q] == 'W':
                count['4'] += 1
            if s[q] == 'R' or s[q] == 'K':
                count['5'] += 1
            if s[q] == 'D' or s[q] == 'E':
                count['6'] += 1
            if s[q] == 'C':
                count['7'] += 1
        val = list(count.values())  # [ 0,0,0,0,0,0,0]
        key = list(count.keys())  # ['1', '2', '3', '4', '5', '6', '7']
        m = 0
        ind = 0
        for t in range(0, len(val)):  # find maximum value from val
            if m < val[t]:
                m = val[t]
                ind = t
        m = key[ind]  # m=group number of maximum occuring group alphabets in protein
        for j in range(0, len(s) - k + 1):
            kmer = s[j:j + k]
            c = ''
            for l in range(0, k):
                if kmer[l] not in groups:
                    c += m
                else:
                    c += groups[kmer[l]]
            V[combinations[int(c)]] += 1

    V = V / (len(s) - 1)
    return np.array(V)

#Function to convert proteins into features
def threemerFromSeq(s):
    k = 3
    groups = {'A': '1', 'V': '1', 'G': '1', 'I': '2', 'L': '2', 'F': '2', 'P': '2', 'Y': '3',
              'M': '3', 'T': '3', 'S': '3', 'H': '4', 'N': '4', 'Q': '4', 'W': '4',
              'R': '5', 'K': '5', 'D': '6', 'E': '6', 'C': '7'}
    crossproduct = [''.join(i) for i in product("1234567", repeat=k)]
    for i in range(0, len(crossproduct)): crossproduct[i] = int(crossproduct[i])
    ind = []
    for i in range(0, len(crossproduct)): ind.append(i)
    combinations = dict(zip(crossproduct, ind))

    V = np.zeros(int((math.pow(7, k))))  # defines a vector of 343 length with zero entries
    try:
        for j in range(0, len(s) - k + 1):
            kmer = s[j:j + k]
            c = ''
            for l in range(0, k):
                c += groups[kmer[l]]
                V[combinations[int(c)]] += 1
    except:
        count = {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0}
        for q in range(0, len(s)):
            if s[q] == 'A' or s[q] == 'V' or s[q] == 'G':
                count['1'] += 1
            if s[q] == 'I' or s[q] == 'L' or s[q] == 'F' or s[q] == 'P':
                count['2'] += 1
            if s[q] == 'Y' or s[q] == 'M' or s[q] == 'T' or s[q] == 'S':
                count['3'] += 1
            if s[q] == 'H' or s[q] == 'N' or s[q] == 'Q' or s[q] == 'W':
                count['4'] += 1
            if s[q] == 'R' or s[q] == 'K':
                count['5'] += 1
            if s[q] == 'D' or s[q] == 'E':
                count['6'] += 1
            if s[q] == 'C':
                count['7'] += 1
        val = list(count.values())  # [ 0,0,0,0,0,0,0]
        key = list(count.keys())  # ['1', '2', '3', '4', '5', '6', '7']
        m = 0
        ind = 0
        for t in range(0, len(val)):  # find maximum value from val
            if m < val[t]:
                m = val[t]
                ind = t
        m = key[ind]  # m=group number of maximum occuring group alphabets in protein
        for j in range(0, len(s) - k + 1):
            kmer = s[j:j + k]
            c = ''
            for l in range(0, k):
                if kmer[l] not in groups:
                    c += m
                else:
                    c += groups[kmer[l]]
            V[combinations[int(c)]] += 1

    V = V / (len(s) - 1)
    return np.array(V)

#Function to convert proteins into features
def prot_feats_seq(seq):
    aa = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    f = []
    X = ProteinAnalysis(str(seq))
    X.molecular_weight()  # throws an error if 'X' in sequence. we skip such sequences
    p = X.get_amino_acids_percent()
    dp = []
    for a in aa:
        dp.append(p[a])
    dp = np.array(dp)
    dp = normalize(np.atleast_2d(dp), norm='l2', copy=True, axis=1, return_norm=False)
    f.extend(dp[0])
    tm = np.array(twomerFromSeq(str(seq)))
    tm = normalize(np.atleast_2d(tm), norm='l2', copy=True, axis=1, return_norm=False)
    f.extend(tm[0])
    thm = np.array(threemerFromSeq(str(seq)))
    thm = normalize(np.atleast_2d(thm), norm='l2', copy=True, axis=1, return_norm=False)
    f.extend(thm[0])
    return np.array(f)

ratio=5

if __name__ == '__main__':
   #Read in PROTACs data
    test_P = []
    test_C = []
    test_Y = []
    wCseq = []
    wPseq = []
    P_uniprot = []
    C_id= []
    with open("prot.csv", "r") as g:
        protacs_positive = g.readlines()
        
        for d in tqdm(protacs_positive):
            uniprot_id, p, compound_id, c = d.strip().split(",")
            try:
                # Convert protein and compounds into features
                xc = getFP(c)
                xp = prot_feats_seq(p)
            except Exception as e:
                print(e)
                continue
            wCseq.append(c)
            wPseq.append(p)
            test_C.append(xc)
            test_P.append(xp)
            test_Y.append(1.0)
            P_uniprot.append(uniprot_id)
            C_id.append(compound_id)
    wY = np.array(test_Y)
    wC = np.array(test_C);
    wP = np.array(test_P);
    wCseq = np.array(wCseq)
    wPseq = np.array(wPseq)

    protacs_protein_set = set(wPseq)
    
    # READ IN ORIGINAL DATA
    with open('./data.txt') as f:  # ../../celegans/original
        D = f.readlines()

    data_C = [];
    data_P = [];
    data_Y = [];
    data_Cseq = [];
    data_Pseq = []
    for d in tqdm(D):
        c, p, y = d.split()
        try:
            #get features
            xc = getFP(c)
            xp = prot_feats_seq(p)
            data_Cseq.append(c)
            data_Pseq.append(p)
            data_C.append(xc)
            data_P.append(xp)
            data_Y.append(float(y))
        except Exception as e:
            print(e)
            continue
    data_Y = np.array(data_Y)
    data_C = np.array(data_C);
    data_P = np.array(data_P);

    # Generating negative training examples (only from original dataset)
    regenerate = False
    if regenerate:
        Pset = list(set(data_Pseq))  # set of protein sequences
        pidx = list(range(len(Pset)))
        Pdict = dict(zip(Pset, pidx))  # seq->index
        Cset = list(set(data_Cseq))  # set of compound sequences
        cidx = list(range(len(Cset)))  #
        Cdict = dict(zip(Cset, cidx))  # str->index
        Epairs = np.array([(Pdict[p], Cdict[c]) for (p, c) in zip(data_Pseq, data_Cseq)])  # dict of pairs
        pos1, negs = Epairs[data_Y == 1, :], Epairs[data_Y != 1, :]
        
        # if the negs are to be sampled such that the both the protein and compound occur in the positive set as well
        # pidx,cidx = list(set(pos[:,0])),list(set(pos[:,1]))
        # pos, negs = list(map(tuple,pos.tolist())),list(map(tuple,negs.tolist())) #
        # below - remove 100% redundant positive and negative examples -- original redundant examples are not removed otherwise
        pos, negs = list(set(map(tuple, pos1.tolist()))), list(set(map(tuple, negs.tolist())))
        NN =  ratio * len(pos)
        negs = []  # comment to use the set of original negatives
        Lnegs = len(negs)
        while len(negs) < NN:
            possible = (random.choice(pidx), random.choice(
                cidx))  # (pidx[np.random.randint(0,len(ppos))],cidx[np.random.randint(0,len(cpos))])
            if possible not in pos and possible not in negs:
                negs.append(possible)
        print('Added Negatives', len(negs) - Lnegs)
        iPdict = {v: k for k, v in Pdict.items()}
        iCdict = {v: k for k, v in Cdict.items()}
    
        data_C = [];
        data_P = [];
        data_Y = [];
        data_Cseq = [];
        data_Pseq = []
        for i, (p, c) in tqdm(enumerate(pos + negs)):
            p = iPdict[p]
            c = iCdict[c]
            try:
                xc = getFP(c)
                xp = prot_feats_seq(p)
            except Exception as e:
                print(e)
                continue
            data_Cseq.append(c)
            data_Pseq.append(p)
            data_C.append(xc)
            data_P.append(xp)
            data_Y.append(int((i < len(pos))))

    data_Y = np.array(data_Y);
    data_C = np.array(data_C);
    data_P = np.array(data_P);

   #%%
    # split protacs data into 5 folds, 4 for training, 1 for validation
    skf = KFold(n_splits=5, shuffle=True)
    unique_proteins = np.asarray(list(set(wPseq)))
    splits = skf.split(unique_proteins)
    
    all_ranks=[]
    count=int(0)
    
    prot_dict={}
    
    for k in unique_proteins:
        prot_dict[k] = [i for i, x in enumerate(wPseq) if str(x) == str(k)]
        
    percentiles = []
    models = []
    mc = keras.callbacks.ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', save_best_only=True)

    for unique_train_idx, unique_test_idx in splits:
        
        # Split unique proteins from protacs dataset into train and test
        train_Pseq = unique_proteins[unique_train_idx]
        test_Pseq = unique_proteins[unique_test_idx]
        train_idx = []
        test_idx = []
        #train data
        for protein in train_Pseq:
            protein_idx = prot_dict[protein]
            for p_idx in protein_idx:
                train_idx.append(p_idx)     
        #validation data                                     
        for protein in test_Pseq:
            protein_idx = prot_dict[protein]
            for p_idx in protein_idx:
                test_idx.append(p_idx)
                
                
                # 4 TRAINING FOLDS OF PROTACS PROTEINS
        count+=1;
        print("\n fold: ", str(count))
        w_P = wP[train_idx]
        w_C = wC[train_idx]
        w_Y = wY[train_idx] # train
        w_Pseq = wPseq[train_idx] # to train
        w_Cseq = wCseq[train_idx] # to train
        
        
                # 1 VALIDATION FOLD OF PROTACS PROTEINS
        test_P = wP[test_idx]
        test_C = wC[test_idx]
        test_Y = wY[test_idx] # test 
        
        
                # COMBINE PROTACS TRAINING FOLDS WITH ORIGINAL TRAINING DATASET
        P = np.concatenate((data_P, w_P), axis=0)
        C = np.concatenate((data_C, w_C), axis=0)
        Y = np.concatenate((data_Y, w_Y), axis=0)
        Pseq = np.concatenate((data_Pseq, w_Pseq), axis=0)
        Cseq = np.concatenate((data_Cseq, w_Cseq), axis=0)
        
        # SCALE TRAIN DATA
        y_train, y_val = Y, test_Y
        Ptr, Ctr = P, C
        Pscaler = StandardScaler().fit(Ptr)
        Cscaler = StandardScaler().fit(Ctr)
        Ptr, Ctr = Pscaler.transform(Ptr), Cscaler.transform(Ctr)
        
        Xtr = np.column_stack((Ptr, Ctr))
        
        xlen = len(Xtr[0])
        
        #Define NN architecture
        model = Sequential()
        
        model.add(Dense(2*xlen, input_dim=xlen, activation='relu', kernel_regularizer=(keras.regularizers.l2(0.001))))
        model.add(Dropout(0.3))
        model.add(Dense(512, activation ='relu', kernel_regularizer=(keras.regularizers.l2(0.001))))
        model.add(Dropout(0.3))
        model.add(Dense(128, activation ='relu', kernel_regularizer=(keras.regularizers.l2(0.001))))
        model.add(Dropout(0.3))
        model.add(Dense(64, activation ='relu', kernel_regularizer=(keras.regularizers.l2(0.001))))
        model.add(Dropout(0.3))
        model.add(Dense(1, activation ='sigmoid', kernel_regularizer=(keras.regularizers.l2(0.001))))

    	# Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        ranks = []
        Ptt, Ctt = Pscaler.transform(test_P), Cscaler.transform(test_C)
        Xtt = np.column_stack((Ptt, Ctt)).astype('float32')
        y_test = test_Y.astype('float32')
        
        model.fit(Xtr, y_train, validation_data = (Xtt, y_test), callbacks = [mc], epochs = 5)
        models.append(model)

#%%
# The following code block calculates the probability of binding for all 
#  (from human proteome) pairs and then stores these values in a csv file.

# The same data preprocessing, e.g. feature extraction and scaling is carried 
#   out on the data before probabilities are calculated.

# The csv is saved to outoput_name

output_name = 'test_proteome.csv'
saved_model = models[1] # select best model here

C_id_filled = []
for k in range(len(C_id)):
    if C_id[k]=="":
        C_id_filled.append(wCseq[k])
    else:
        C_id_filled.append(C_id[k])


from Bio import SeqIO

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

input_file = "uniprot-proteome_UP000005640.txt"

fasta_sequences = SeqIO.parse(open(input_file),'fasta')
prot_names = []
prot_sequences = []
#get protein sequences from proteome file
for fasta in fasta_sequences:
    name, sequence = fasta.id, str(fasta.seq)
    prot_names.append(name)
    prot_sequences.append(sequence)

with open(output_name, 'ab') as f:
    
    prot_sequences = prot_sequences
    for k in range(len(prot_sequences)):
        protein_seq = prot_sequences[k]
        try:
            array_protein_seq = prot_feats_seq(protein_seq)
            
            unique_Cseq = list(set(wCseq))
            array_unique_Cseq = []
            for comp in unique_Cseq:
                    array_unique_Cseq.append(getFP(comp))
            
            one_human_prot = []
            for comp in array_unique_Cseq:
                one_human_prot.append((array_protein_seq, comp))
            
            one_human_prot = np.asarray(one_human_prot)
            
            Ps = np.asarray([np.expand_dims(x, axis=1) for x in one_human_prot[:, 0]])[:, :, 0]
            Cs = np.asarray([np.expand_dims(x, axis=1) for x in one_human_prot[:, 1]])[:, :, 0]
            
            Ps, Cs = Pscaler.transform(Ps), Cscaler.transform(Cs)
            Xtest = np.column_stack((Ps, Cs))
            
            result = saved_model.predict(Xtest)
            
            protein_name = prot_names[k].split("|")[1]
            uniprot_id_column = np.asarray([protein_name for x in range(len(Xtest))])
            
            comp_id_column = np.asarray(list(set(C_id_filled)))
            
            csv_output = np.column_stack((uniprot_id_column, comp_id_column, result))
            
            if k==0:
                np.savetxt(f, csv_output, fmt="%s", delimiter=',', header='Uniprot_ID,Compound_Name,Binding_Prediction')
            else:
                np.savetxt(f, csv_output, fmt="%s", delimiter=',')
        except Exception as e:
            print(e)
            continue
