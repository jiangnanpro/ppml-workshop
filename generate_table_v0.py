import numpy as np
import csv
import random
import pickle
import tensorflow as tf
from tensorflow.keras import layers
from tqdm import tqdm

import warnings

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split

from sklearn import metrics
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import euclidean_distances

pickle_file = './data/QMNIST_tabular_ppml.pickle'

with open(pickle_file, 'rb') as f:
  pickle_data = pickle.load(f)
  x_defender = pickle_data['x_defender']
  x_reserve = pickle_data['x_reserve']
  y_defender = pickle_data['y_defender']
  y_reserve = pickle_data['y_reserve']
  del pickle_data
print('Data loaded.')

NUM_CLASSES = 10

y_defender = y_defender[:,0]
y_reserve = y_reserve[:,0]


from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KDTree
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

def defender_model_fn(random_seed, number_model = 0):

    if number_model == 0:
      model = LogisticRegression(random_state=0)
    elif number_model == 1:
      model = RidgeClassifier(random_state=0)
    elif number_model == 2:
      model = LinearSVC(random_state=0)
    elif number_model == 3:
      model = DecisionTreeClassifier(random_state=random_seed)
    elif number_model == 4:
      model = KNeighborsClassifier() # no random_state
    elif number_model == 5:
      model = KNeighborsClassifier(algorithm='kd_tree') # no random_state
    elif number_model == 6:
      model = GaussianNB() # no random_state
    elif number_model == 7:
      model = SVC(probability=True,random_state=0)
    elif number_model == 8:
      model = SGDClassifier(random_state=0)
    elif number_model == 9:
      model = Perceptron(random_state=0)
    elif number_model == 10:
      model = RandomForestClassifier(random_state=0)
    elif number_model == 11:
      model = MLPClassifier(random_state=0)
    else:
      raise Exception("Invalid number of model!", number_model)
       
    return model

# if random_select, randomly extract two points from both defender & reserve dataset, and repeat for n_extract times
random_select = True
n_extract = 100


def create_mock_defender_models(defender, data_in, data_out, n_records = 48, random_select = True, n_extract = 100, given_index = False, random_seed = None, number_model = 0):
    
    difference_in = []
    difference_out = []
    
    number_loop = 0
    
    if random_select == True:
        number_loop = n_extract
    else:
        number_loop = data_in[0].shape[0]
        
    for i in range(number_loop):

        if random_select == True:
            index = random.randint(0,n_records-1)
        else:
            index = i

        evaluation_data_in = data_in[0][index]
        evaluation_label_in = data_in[1][index]

        evaluation_data_out = data_out[0][index]
        evaluation_label_out = data_out[1][index]

        evaluation_data = np.array([evaluation_data_in, evaluation_data_out])
        evaluation_label = np.array([evaluation_label_in, evaluation_label_out])

        evaluation = evaluation_data, evaluation_label


        attack_train_data_in = np.delete(data_in[0], index, axis=0)
        attack_train_label_in = np.delete(data_in[1], index, axis=0)

        attack_in = attack_train_data_in, attack_train_label_in


        attack_train_data_out = np.delete(data_out[0], index, axis=0)
        attack_train_label_out = np.delete(data_out[1], index, axis=0)

        attack_out = attack_train_data_out, attack_train_label_out

        
        if given_index == True:

            attack_in_plus_one_in = np.insert(attack_in[0], index, evaluation[0][0].reshape(1,attack_in[0].shape[1]), axis=0), np.insert(attack_in[1], index, evaluation[1][0], axis=0)
            attack_in_plus_one_out = np.insert(attack_in[0], index, evaluation[0][1].reshape(1,attack_in[0].shape[1]), axis=0), np.insert(attack_in[1], index, evaluation[1][1], axis=0)

        else:

            attack_in_plus_one_in = np.vstack((evaluation[0][0].reshape(1,attack_in[0].shape[1]),attack_in[0])), np.hstack(( evaluation[1][0],attack_in[1]))
            attack_in_plus_one_out = np.vstack((evaluation[0][1].reshape(1,attack_in[0].shape[1]),attack_in[0])), np.hstack(( evaluation[1][1],attack_in[1]))

        
        M_cD_in = defender_model_fn(random_seed=random_seed, number_model=number_model)
        M_cD_out = defender_model_fn(random_seed=random_seed, number_model=number_model)

        M_cD_in.fit(attack_in_plus_one_in[0], attack_in_plus_one_in[1])
        M_cD_out.fit(attack_in_plus_one_out[0], attack_in_plus_one_out[1])
        

        try:
            predict = defender.decision_function(attack_in[0])
            M_cD_in_predict = M_cD_in.decision_function(attack_in[0])
            M_cD_out_predict = M_cD_out.decision_function(attack_in[0])

        except AttributeError:
            #print('No decision_function(), using predict_proba()')
            predict = defender.predict_proba(attack_in[0])
            M_cD_in_predict = M_cD_in.predict_proba(attack_in[0])
            M_cD_out_predict = M_cD_out.predict_proba(attack_in[0])

        diff_in = np.mean(np.linalg.norm(M_cD_in_predict-predict))
        diff_out = np.mean(np.linalg.norm(M_cD_out_predict-predict))

        difference_in.append(diff_in)
        difference_out.append(diff_out)
    
    return difference_in, difference_out
    


def compute_utility_privacy(random_seed = None, number_model = 0, given_index = False):

    #if given_index, the attack will put the two left out points back to their original place
    
    if random_seed != None:
      random_seeds = random.sample(range(0,100), 2)
      np.random.seed(random_seeds[0])
      random.seed(random_seeds[1])

    accuracy_in_all = []
    accuracy_out_all = []

    difference_in_all = []
    difference_out_all = []

    n_records = 1600
    n_trials = 3

    for i in range(n_trials):
        
        random_indexes = random.sample(range(0,200000), n_records)
        data_in = x_defender[random_indexes], y_defender[random_indexes]
        data_out = x_reserve[random_indexes], y_reserve[random_indexes]
        
        defender_model = defender_model_fn(random_seed=random_seed, number_model=number_model)
        defender_model.fit(data_in[0],data_in[1])
        
        predict_in = defender_model.predict(data_in[0])
        acc_in = accuracy_score(data_in[1], predict_in)
        
        predict_out = defender_model.predict(data_out[0])
        acc_out = accuracy_score(data_out[1], predict_out)
        
        accuracy_in_all.append(acc_in)
        accuracy_out_all.append(acc_out)
        
        
        difference_in, difference_out = create_mock_defender_models(defender = defender_model,
                                        data_in = data_in,
                                        data_out = data_out,
                                        n_records = n_records,
                                        random_select = random_select,
                                        n_extract = n_extract,
                                        given_index = given_index,
                                        random_seed = random_seed,
                                        number_model = number_model)
        
        difference_in_all.append(difference_in)
        difference_out_all.append(difference_out)


    utility_in_all = (np.array(accuracy_in_all)*10-1)/9
    utility_out_all = (np.array(accuracy_out_all)*10-1)/9


    # compute the privacy values by comparing all model pairs

    privacy_all = []
    variance_all = []
    sigma_error_all = []

    for i in range(len(difference_in_all)):
        
        difference_in = difference_in_all[i]
        difference_out = difference_out_all[i]
        
        results = []

        for j in range(len(difference_in)):
            for k in range(len(difference_out)):
                
                if difference_in[j] == difference_out[k]:
                    random_number = random.randint(0,99)
                    if random_number < 50:
                        results.append(1)
                    else:
                        results.append(0)

                elif difference_in[j] < difference_out[k]:
                    results.append(1)
                else:
                    results.append(0)

        n = len(results)
        p = 1-np.sum(results)/n

        privacy = min(2*p,1)
        variance = 2*p*(1-p)/n
        sigma_error = np.sqrt(p*(1-p)/(2*n))
        
        privacy_all.append(privacy)
        variance_all.append(variance)
        sigma_error_all.append(sigma_error)
    
    #print('utility_in:',np.mean(utility_in_all))
    #print('utility_out:',np.mean(utility_out_all))
    #print('privacy:',np.mean(privacy_all))
    #print('variance:',np.mean(variance_all))
    #print('sigma_error:',np.mean(sigma_error_all))

    u_in = np.mean(utility_in_all)
    u_out = np.mean(utility_out_all)
    pr = np.mean(privacy_all)
    v = np.mean(variance_all)
    s = np.mean(sigma_error_all)

    return u_in,u_out,pr,v,s
    

results = []

for n in tqdm(range(12)):
  
  u_in,u_out,pr,v,s = compute_utility_privacy(random_seed = 0, number_model = n, given_index = True)
  results.append([n,0,u_in,u_out,pr,v,s])
  u_in,u_out,pr,v,s = compute_utility_privacy(random_seed = 0, number_model = n, given_index = False)
  results.append([n,1,u_in,u_out,pr,v,s])
  u_in,u_out,pr,v,s = compute_utility_privacy(random_seed = None, number_model = n, given_index = False)
  results.append([n,2,u_in,u_out,pr,v,s])


with open('./results/results_table.csv','w',newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Number_model','Level_randomness','Utility_in','Utility_out','Privacy','Variance','Sigma_error'])
    writer.writerows(results)
