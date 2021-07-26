# -*- coding: utf-8 -*-
"""
Created on Sun July 25, 2021

@author: Joe
"""

import numpy as np
import os # for reading files in directory
import pickle as pkl # for dumping weights

# Python script to train a neural net with 2 hidden layers
#   to classify MNIST images into 0..9

# Problem constants
N = 400 # size of trng/test sets
K = 10 # number of classes

###############################
# load the data               #
###############################

print("Loading data...")

current_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(current_dir,'x_defender.pickle'), 'rb') as f:
    x_defender = pkl.load(f)
  
with open(os.path.join(current_dir,'x_reserve.pickle'), 'rb') as f:
    x_reserve = pkl.load(f)
  
with open(os.path.join(current_dir,'y_defender_flipped.pickle'), 'rb') as f:
    y_defender = pkl.load(f)
  
with open(os.path.join(current_dir,'y_reserve_flipped.pickle'), 'rb') as f:
    y_reserve = pkl.load(f)
    
print("Data loaded.")
  
 
trng_array = x_defender[:N,:]
test_array = x_reserve[:N,:]
y_trng = y_defender[:N]
y_test = y_reserve[:N]

y_trng_ind = np.argmax(y_trng, axis=1)
y_test_ind = np.argmax(y_test, axis=1)

###############################
# start learning              #
###############################
np.random.seed(42)

# hyper-parameters
BATCH_SIZE = 40 # batch size hyper-parameter
BATS_PER_EPOCH = N//BATCH_SIZE # number of batches per EPOCH
EPOCHS = 1001 # numer of epochs to try (check for optimal capacity)
ETA = 0.003 # learning rate hyper-parameter
LAMDA = 0.000001 # regularization hyper-parameter NOT USED THIS ASSIGNMENT
N0 = 512
N1 = 128 # num nodes in H1

# allocate memory for variables
H1 = np.zeros((N1, BATCH_SIZE)) # hidden nodes
Yhat = np.zeros((K, BATCH_SIZE)) # Yhat
pdRelu1 = np.zeros((BATCH_SIZE, N1, N1)) # partial relu1(z)/partial z
pdSigma = np.zeros((BATCH_SIZE, K, K)) # partial softmax(z)/partial z
W1 = np.random.uniform(low=-0.001, high=0.001, size=(N0, N1)) # initialize weights
W2 = np.random.uniform(low=-0.001, high=0.001, size=(N1, K))
W10 = np.zeros((N1, 1)) # initialize bias
W20 = np.zeros((K, 1))
gradH1 = np.zeros((BATCH_SIZE, N1, 1)) # gradients
gradYhat = np.zeros((K, BATCH_SIZE))
gradW1 = np.zeros((BATCH_SIZE, N0, N1))
gradW2 = np.zeros((BATCH_SIZE, N1, K))
gradW10 = np.zeros((BATCH_SIZE, N1, 1))
gradW20 = np.zeros((BATCH_SIZE, K, 1))

# create arrays to store progress
NUM_BATCHES = BATS_PER_EPOCH * EPOCHS # total number of batches
norms = np.zeros(NUM_BATCHES) # norm of gradient for each mini-batch
trng_loss = np.zeros(EPOCHS) # loss on entire training set each epoch
test_loss = np.zeros(EPOCHS) # loss on entire test set each epoch
trng_accuracy = np.zeros(EPOCHS) # accuracy on entire training set
test_accuracy = np.zeros(EPOCHS) # accuracy on entire test set

best_ind = 0 # index of best test_accuracy
best_test = 0.40 # value of best test_accuracy

# defining my activation functions
def relu(x):
    '''the Relu activation function'''
    return np.maximum(0,x)

def softmax(x):
    '''the softmax activation function applied to each column'''
    temp = x - np.amax(x, axis=0, keepdims=True) # to avoid overflow
    # softmax(z_k) = exp(z_k)/sum_k' {exp(z_k')}
    return np.exp(temp)/np.sum(np.exp(temp), axis=0, keepdims=True)


# make copy of training data to shuffle each epoch
#     for stochastic gradient descent
epoch_array = np.concatenate((trng_array,y_trng), axis=1)

print("Starting to train...")

for i in range(EPOCHS):
    np.random.shuffle(epoch_array) # shuffle the training data for SGD

    for j in range(BATS_PER_EPOCH):

        # select a mini-batch
        bStart = j*BATCH_SIZE
        bEnd = j*BATCH_SIZE + BATCH_SIZE
        xbatch = np.transpose(epoch_array)[0:(N0),bStart:bEnd]
        ybatch = np.transpose(epoch_array)[(N0):(N0+1+K),bStart:bEnd]
        t = i*BATS_PER_EPOCH + j # index for recording loss/accuracy/etc.

        # FORWARD PASS
        H1 = relu(np.tensordot(np.transpose(W1),xbatch, axes=1) + W10)
        Yhat = softmax(np.tensordot(np.transpose(W2),H1, axes=1) + W20)

        # BACK PROP - compute the gradients, working backwards
        gradYhat = -ybatch/Yhat # gradient of output

        # partial softmax(z)/partial z
        for b in range(BATCH_SIZE):
            pdSigma[b,:,:] = np.matmul(
                np.diag(Yhat[:,b]),
                    np.identity(K) - Yhat[:,b]
                    )

        gradW2 = (
            np.matmul(
                np.matmul(
                    np.expand_dims(np.transpose(H1),axis=2), # col vecs
                    np.expand_dims(np.transpose(gradYhat),axis=1) # row vecs
                    ), # outer product, effectively
                pdSigma
                )
            )

        gradW20 = (
            np.matmul(
                pdSigma,
                np.expand_dims(np.transpose(gradYhat),axis=2), # col vecs
                )
            )

        gradH1 = (
            np.matmul(
                W2,
                np.matmul(
                    pdSigma,
                    np.expand_dims(np.transpose(gradYhat),axis=2), # col vecs
                    )
                )
            )

        # partial Relu(z)/partial z for H1
        for b in range(BATCH_SIZE):
            pdRelu1[b,:,:] = np.diag(np.sign(H1[:,b]))

        gradW1 = (
            np.matmul(
                np.matmul(
                    np.expand_dims(np.transpose(xbatch),axis=2), # col vecs
                    np.swapaxes(gradH1,1,2) # row vecs
                    ), # outer product, effectively
                pdRelu1
                )
            )

        gradW10 = (
            np.matmul(
                pdRelu1,
                gradH1 # col vecs
                )
            )


        # update the weights by subtracting ETA * the gradient of the loss
        W1 = W1 - ETA*np.average(gradW1,axis=0) - LAMDA*W1
        W10 = W10 - ETA*np.average(gradW10,axis=0) - LAMDA*W10
        W2 = W2 - ETA*np.average(gradW2,axis=0) - LAMDA*W2
        W20 = W20 - ETA*np.average(gradW20,axis=0) - LAMDA*W20

        # measure convergence of gradient each batch
        norms[t] = np.sqrt(
            np.linalg.norm(np.average(gradW1,axis=0))**2 +
            np.linalg.norm(np.average(gradW10,axis=0))**2 +
            np.linalg.norm(np.average(gradW2,axis=0))**2 +
            np.linalg.norm(np.average(gradW20,axis=0))**2
            )

        # print progress every 100 mini-batches
        #if t % 2 == 0:
        #    print('Epoch = ' + str(i)
        #          + ', batch = ' + str(j)
        #          + ', grad norm = ' + str(norms[t])
        #          )

    # record loss and accuracy each EPOCH
    # total loss (-LCL) on training set
    trng_XW = (np.tensordot(
                   np.transpose(W2),
                   relu(
                       np.tensordot(
                           np.transpose(W1),
                           np.transpose(trng_array),
                           axes=1
                           )
                       + W10),
                   axes=1
                   )
               + W20)

    trng_loss[i] = (
            -np.sum(np.transpose(y_trng) *
                    (trng_XW -
                     np.log(np.sum(np.exp(trng_XW),
                                   axis=0, keepdims=True)
                            )
                     )
                    )
            )/N

    # total loss (-LCL) on test set
    test_XW = (np.tensordot(
                   np.transpose(W2),
                   relu(
                       np.tensordot(
                           np.transpose(W1),
                           np.transpose(test_array),
                           axes=1
                           )
                        + W10),
                    axes=1
                    )
                + W20)
    
    test_loss[i] = (
            -np.sum(np.transpose(y_test) *
                    (test_XW -
                     np.log(np.sum(np.exp(test_XW),
                                   axis=0, keepdims=True)
                            )
                     )
                    )
            )/N

    # total accuracy on training set and test set
    trng_accuracy[i] = (
            np.sum(y_trng_ind == np.argmax(trng_XW, axis=0))/N)
    test_accuracy[i] = (
        np.sum(y_test_ind == np.argmax(test_XW, axis=0))/N)

    # print loss and accuracy every 5 epochs
    if ((i==4) or (i % 50 == 0)):
        print('Epoch = ' + str(i)
                      + ', trng_loss = ' + str(trng_loss[i])
                      + ', test_loss = ' + str(test_loss[i])
                      + ', \n' + '\t  '
                      + 'trng_accuracy = ' + str(trng_accuracy[i])
                      + ', test_accuracy = ' + str(test_accuracy[i])
                      )
        
    if (i == 4):
        W1_early = W1.copy()
        W10_early = W10.copy()
        W2_early = W2.copy()
        W20_early = W20.copy()

    # if test accuracy has improved, save weights as new optimal
    if (test_accuracy[i] > best_test):
        best_ind = i
        best_test = test_accuracy[best_ind].copy()
        W1star = W1.copy()
        W10star = W10.copy()
        W2star = W2.copy()
        W20star = W20.copy()

    W1_last = W1.copy()
    W10_last = W10.copy()
    W2_last = W2.copy()
    W20_last = W20.copy()

###############################
# save the results            #
###############################

print("Saving optimal weights...")

dump_dict = dict()
dump_dict['W1_early'] = W1star
dump_dict['W10_early'] = np.ndarray.flatten(W10star)
dump_dict['W2_early'] = W2star
dump_dict['W20_early'] = np.ndarray.flatten(W20star)
dump_dict['W1star'] = W1star
dump_dict['W10star'] = np.ndarray.flatten(W10star)
dump_dict['W2star'] = W2star
dump_dict['W20star'] = np.ndarray.flatten(W20star)
dump_dict['W1_last'] = W1star
dump_dict['W10_last'] = np.ndarray.flatten(W10star)
dump_dict['W2_last'] = W2star
dump_dict['W20_last'] = np.ndarray.flatten(W20star)
dump_dict['EPOCHS'] = EPOCHS
dump_dict['norms'] = norms
dump_dict['trng_loss'] = trng_loss
dump_dict['test_loss'] = test_loss
dump_dict['best_ind'] = best_ind

# save the weights from the optimal capacity model
filehandler = open(os.path.join(current_dir,"nn_dict.pickle"),"wb")
pkl.dump(dump_dict,
            filehandler)
filehandler.close()

###############################
# plot the results            #
###############################

print("Plotting the results...")

print("but not really.")

print("All finished.")