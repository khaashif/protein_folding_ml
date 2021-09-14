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

def getFP(s, r=3, nBits=1024):
    # Generate chemical compound features from smiles string
    compound = Chem.MolFromSmiles(s.strip())
    fp = AllChem.GetMorganFingerprintAsBitVect(compound, r, nBits=nBits)
    # fp = pat.GetAvalonCountFP(compound,nBits=nBits)
    m = np.zeros((0,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, m)
    return m


def twomerFromSeq(s):
    # Function used to convert amino acid string into feature array
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


def threemerFromSeq(s):
    # Function used to convert amino acid string into feature array
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


def prot_feats_seq(seq):
    # Function used to convert amino acid string into feature array
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


# Define ratio to use when regen=True
ratio = 5

if __name__ == '__main__':
  #READ IN PROTACS DATA
    test_P = []
    test_C = []
    test_Y = []
    wCseq = []
    wPseq = []
    
    with open("protacs.csv", "r") as g:
        protacs_positive = g.readlines()
        
        for d in tqdm(protacs_positive):
            c, p, y = d.split(" ")
            try:
                #convert strings into feature arrays
                xc = getFP(c)
                xp = prot_feats_seq(p)
            except Exception as e:
                print(e)
                continue
            wCseq.append(c)
            wPseq.append(p)
            test_C.append(xc)
            test_P.append(xp)
            test_Y.append(float(y))
    wY = np.array(test_Y)
    wC = np.array(test_C);
    wP = np.array(test_P);
    wCseq = np.array(wCseq)
    wPseq = np.array(wPseq)

    protacs_protein_set = set(wPseq)
    
    # READ IN ORIGINAL DATA
    with open('./data.txt') as f:
        D = f.readlines()

    data_C = [];
    data_P = [];
    data_Y = [];
    data_Cseq = [];
    data_Pseq = []
    for d in tqdm(D):
        c, p, y = d.split()
        try:
            if p not in protacs_protein_set: # exclude 106 shared proteins
                #convert strings into feature arrays
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
    # Converts FDA approved compounds into numerical arrays to be used to 
    #    generate negative interactions to calculate RFPP score later
    converted_compounds = []
    with open("approved_compounds.csv", "r") as f:
        app_comps = f.readlines()
    
    for c in app_comps:
        c_1 = c.replace(" ", "")
        c_1 = c_1.replace("\n", "")
        try:
            xc = getFP(c_1)
            converted_compounds.append(xc)
        except Exception as e:
            print(e)
            continue
   
   #%%
    # Split PROTACs data 5 ways to add 4 folds to training and 1 fold for testing
    skf = KFold(n_splits=5, shuffle=True)
    unique_proteins = np.asarray(list(set(wPseq)))
    splits = skf.split(unique_proteins)
    
    all_ranks=[]
    count=int(0)
    
    prot_dict={}
    
    #dict of all positive examples - unique proteins against corresponding PROTAC compounds 
    for k in unique_proteins:
        prot_dict[k] = [i for i, x in enumerate(wPseq) if str(x) == str(k)]
        
    percentiles = []
    
    for unique_train_idx, unique_test_idx in splits:
        
        # Split unique proteins from protacs dataset into train and test
        train_Pseq = unique_proteins[unique_train_idx]
        test_Pseq = unique_proteins[unique_test_idx]
        train_idx = []
        test_idx = []
        
        for protein in train_Pseq:
            protein_idx = prot_dict[protein]
            for p_idx in protein_idx:
                train_idx.append(p_idx)                                          
        for protein in test_Pseq:
            protein_idx = prot_dict[protein]
            for p_idx in protein_idx:
                test_idx.append(p_idx)
                
                
        # Protac Training 4/5 Folds
        count+=1;
        print("\n fold: ", str(count))
        w_P = wP[train_idx]
        w_C = wC[train_idx]
        w_Y = wY[train_idx] # train
        w_Pseq = wPseq[train_idx] # to train
        w_Cseq = wCseq[train_idx] # to train
        
        
        # Protac Test 1/5 Folds
        test_P = wP[test_idx]
        test_C = wC[test_idx]
        test_Y = wY[test_idx] # test 
        
        
        # Add protac training fold to original training data
        P = np.concatenate((data_P, w_P), axis=0)
        C = np.concatenate((data_C, w_C), axis=0)
        Y = np.concatenate((data_Y, w_Y), axis=0)
        Pseq = np.concatenate((data_Pseq, w_Pseq), axis=0)
        Cseq = np.concatenate((data_Cseq, w_Cseq), axis=0)
        
        
        
        y_train, y_val = Y, test_Y
        Ptr, Ctr = P, C
        # Scale Training Data
        Pscaler = StandardScaler().fit(Ptr)
        Cscaler = StandardScaler().fit(Ctr)
        Ptr, Ctr = Pscaler.transform(Ptr), Cscaler.transform(Ctr)
        
        Xtr = np.column_stack((Ptr, Ctr)) # stack protein and compound features
        
        xlen = len(Xtr[0])
        
        # Define NN Model
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
        #Fit training data
        model.fit(Xtr, y_train)
        
        scores=[]
        ranks = []
        
        test_Pseq = wPseq[test_idx]
        protein_set = list(set(test_Pseq))
        
        ## For loop through each unique protein in test set and generate RFPP scores:
        for k in range(len(protein_set)):
            test_set = []
            protein = protein_set[k]
            protein_idx = [i for i, x in enumerate(test_Pseq) if x == protein] # contains all of the same proteins
            
            #positive interactions for given protein
            for j in protein_idx:
                positive = [test_P[j], test_C[j], 1]
                test_set.append(positive)
            #generate nnegative interactions for given protein
            #combine protein with FDA approved compounds to create negative examples
            for l in range(len(converted_compounds)): 
                negative = [test_P[k], converted_compounds[l], 0]
                test_set.append(negative)
                
            # Test the dataset that we have just generated and obtain likelihood scores
            test_set = np.asarray(test_set, dtype=object)
            Ptest = np.asarray([np.expand_dims(x, axis=1) for x in test_set[:, 0]])[:, :, 0]
            Ctest = np.asarray([np.expand_dims(x, axis=1) for x in test_set[:, 1]])[:, :, 0]
            Ytest = test_set[:, 2]
            
            # Scale test data
            y_test = Ytest
            w_Ptt, w_Ctt = Pscaler.transform(Ptest), Cscaler.transform(Ctest)
            
            w_Ptt = list(w_Ptt)
            w_Ctt = list(w_Ctt)

            y_test = np.asarray(y_test)
            w_Ptt = np.asarray(w_Ptt)
            w_Ctt = np.asarray(w_Ctt)
            
            Xtt = np.column_stack((w_Ptt, w_Ctt))
            
            # Predict binding likelihoods and obtain rank metric:
            score = model.predict(Xtt) # calculate likelihoods
            binding_probability = score[:, 0] # takes all binding likelihoods
            
            # obtain rank of example with highest binding likelihood
            idx_array = np.asarray([x for x in range(len(binding_probability))])
            # Sort probabilities
            sorted_probs = [x for _,x in sorted(zip(binding_probability, idx_array), reverse=True)]
            
            # append ranks into list
            protein_ranks = []
            for x in range(len(protein_idx)):
                protein_ranks.append(list(sorted_probs).index(x))

            min_rank = min(protein_ranks) # = RFPP            
            ranks.append(min_rank)
            all_ranks.append(min_rank)
            
            scores.append(score)
        rank_percentiles = [1, 5, 10, 30, 40, 50, 60, 70, 80, 90, 95, 99, 100]
        percentiles.append(np.percentile(ranks, rank_percentiles))
        
    # mean_p is the average RFPP percentile scores of all train/test splits 
    mean_p = [np.mean(col) for col in zip(*percentiles)]
    #mean_p is the value we would plot
    
    
    print("\n Avg Median rank", mean_p[5])
