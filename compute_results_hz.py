

import os
import numpy as np
from sklearn.metrics import roc_auc_score 
import argparse
import scipy.stats
import pandas as pd


def compute_mean_and_confidence_interval(x, confidence=0.95):
    """
    returns the mean and the confidence interval, which are two real numbers

    x: iterable

    high - mean_ == mean_ - low except some numerical errors in rare cases
    """
    mean_ = np.mean(x)
    low, high = scipy.stats.t.interval(confidence, len(x) - 1, loc=mean_, scale=scipy.stats.sem(x))
    return mean_, high - mean_

def compute_AUC_score(yhat_all, oneStep_yhat_all, df, N, M):
    distances = - np.sum( yhat_all * np.log( oneStep_yhat_all ), axis = 1 )
    auc_list = []
    for seed in np.array(list(range(10))) + 10:
        np.random.seed(seed)

        mask1_D = np.random.choice(np.arange(0,N), size=M)
        mask2_D = np.random.choice(np.arange(0,N), size=M)
        mask1_R = np.random.choice(np.arange(N,2*N), size=M)
        mask2_R = np.random.choice(np.arange(N,2*N), size=M)

        resample_mem_lbls = np.concatenate( ( np.ones((1,M)), np.zeros((1,M)) ), axis=1 )

        distance_diff = np.concatenate(( distances[mask1_R] - distances[mask1_D], distances[mask2_D] - distances[mask2_R]), axis = 0)

        try:
            auc = roc_auc_score(resample_mem_lbls[0,:], distance_diff)
            auc_list.append(auc)
        except ValueError as e:
            pass

    if len(auc_list) > 0:
        auc_average, auc_confidence_interval = compute_mean_and_confidence_interval(auc_list)
        df.append((os.path.basename(folder_name), auc_average, auc_confidence_interval))
    return df


# cross entropy between probability vectors
def cross_ent1(p, q):
  
    return -1*(p * np.log(q)).sum(axis=0)


# instead of probabilities 'q', it uses the outputs before softmax
def cross_ent2(p, xw): 
    
    colmaxes = np.amax(xw, axis=0, keepdims=True) # to avoid overflow
  
    log_q = np.log( np.sum( np.exp(xw - colmaxes), axis=0 ) ) - xw + colmaxes
  
    return np.sum( p * log_q, axis=0 )


# cross entropy with buffer, in case any probabilities in q are 0 or 1
def cross_ent3(p, q):
    
    if (np.min(q)<0 or np.max(q)>1):
        raise ValueError("Array has values not in [0,1]")
        
    # to avoid errors in logs for q=0 or q=1
    if np.min(q)==0:
        low = np.min(q[q>0])
        q += low/(1e12) # adding a small positive value 
        
    if np.max(q) >= 1:
        high = np.max(q)
        q /= (high*(1+1e-15)) # dividing by number slightly larger than 1
  
    return -1*(p * np.log(q)).sum(axis=0)



# estimates the accuracy A_ltu, based on a sample of M pairs
def A_ltu(yhat_all, OneStep_yhat_all, fun=3, N=None, Nb=None, M=2500):
    
    # Use if all probabilities are strictly between 0 and 1
    if fun == 1:
        dist_to_Mi_Ui = cross_ent1(yhat_all.T, OneStep_yhat_all.T)
    
    # Use if OneStep_yhat_all is outputs w/o softmax, instead of probabilities
    if fun == 2:
        dist_to_Mi_Ui = cross_ent2(yhat_all.T, OneStep_yhat_all.T)
    
    # Use if some probabilities are exactly 0 or 1
    if fun == 3:
        dist_to_Mi_Ui = cross_ent3(yhat_all.T, OneStep_yhat_all.T)

    # Number of pairs to sample
    #M = 2500
    
    # By default, the first half of the data are "defender"
    if not N:
        N = yhat_all.shape[0]//2
        
    # The rest are "reserved"
    if not Nb:
        Nb = yhat_all.shape[0] - N
        
    if (N + Nb) != yhat_all.shape[0]:
        print(N)
        print(Nb)
        print(yhat_all.shape[0])
        raise ValueError("N + Nb does not equal the size of yhat_all")
        
    if yhat_all.shape != OneStep_yhat_all.shape:
        raise ValueError("The shapes of yhat_all and OneStep_yhat_all should be the same")

    # Masks of randomly chosen points
    mask1_D = np.random.choice(N, size=M)
    mask1_R = np.random.choice(Nb, size=M) + N

    # Accuracy is proportion for which the "defender" have smaller distances
    acc_ltu = (np.mean(dist_to_Mi_Ui[mask1_D] < dist_to_Mi_Ui[mask1_R])
               + 0.5*np.mean(dist_to_Mi_Ui[mask1_D] == dist_to_Mi_Ui[mask1_R]) )
    
    se = np.sqrt(acc_ltu*(1-acc_ltu)/M)
    
    return acc_ltu, se


def Privacy(yhat_all, oneStep_yhat_all, N=None, Nb=None, M=2500):

    acc_ltu, se = A_ltu(yhat_all, oneStep_yhat_all, fun=3, N=N, Nb=Nb, M=M)
    
    priv = 2*(1 - acc_ltu)
    se = 2*se
    
    return priv, se

def get_se(p, sample_size):
    return np.sqrt(p * (1 - p) / sample_size)

if __name__ == "__main__":

    N = 3000


    folder_names = {"supervised_normal_fc": 97.186, 
                    "supervised_normal_whole": 99.733, 
                    "supervised_long_fc": 97.616, 
                    "supervised_long_whole": 99.727, 
                    "supervised_flipped_fc": 78.223, 
                    "supervised_flipped_whole": 77.828, 
                    "small_fake_mnist__attack_mode_forward_target_domain": 98.514,
                    "large_fake_mnist__attack_mode_forward_target_domain": 98.675,
                    "small_fake_mnist__attack_mode_transfer_loss": 98.514, 
                    "large_fake_mnist__attack_mode_transfer_loss": 98.675,
                    "small_fake_mnist__attack_mode_total_loss": 98.514,
                    "large_fake_mnist__attack_mode_total_loss": 98.675}
    df = []
    for folder_name, reserve_set_classification_accuracy in folder_names.items():

        folder_name = os.path.join("hz_intermediate_results", folder_name)

        with open(os.path.join(folder_name, "yhat_all.npy"), "rb") as f:
            yhat_all = np.load(f)

        with open(os.path.join(folder_name, "oneStep_yhat_all.npy"), "rb") as f:
            oneStep_yhat_all = np.load(f)

        """
        with open(os.path.join(folder_name, "gradNorm_all.npy"), "rb") as f:
            gradNorm_all = np.load(f)
        """
        
        M = int(N / 2)

        #compute_AUC_score(yhat_all, oneStep_yhat_all, df, N, M)

        privacy, privacy_standard_error = Privacy(yhat_all, oneStep_yhat_all, N=N, Nb=N, M=M)
        df.append((os.path.basename(folder_name), reserve_set_classification_accuracy, 
            get_se(reserve_set_classification_accuracy, ), privacy, privacy_standard_error))
        
    df = pd.DataFrame(df, columns=["folder_name", "utility", "utility_standard_error", 
        "privacy", "privacy_standard_error"])
    df.to_csv("hz_table.csv", sep="\t", encoding="utf-8")
    print(df)





