############### Imports #############################

# TODO: insert your project 3 file name here:
import project3 as p3
import numpy as np

############### Read data ############################

X = p3.readData('matrixCompletionTest_incomplete.txt')
Xc = p3.readData('matrixCompletionTest_complete.txt')
K = 4
n, d = np.shape(X)


############### Functions #############################

def initToyFixed(X,K):
    n, d = np.shape(X)
    P=np.ones((K,1))/float(K)
    Mu = np.array([[2.,3.,4.,0.,0.],\
[0.,0.,2.,3.,3.],\
[2.,5.,2.,3.,4.],\
[0.,5.,3.,4.,2.]])
    Var=np.mean( (X-np.tile(np.mean(X,axis=0),(n,1)))**2 )*np.ones((K,1))
    return (Mu,P,Var)


############### Init ##################################

Mu,P,Var = initToyFixed(X,K)


########## Test E step and M step #####################

postE, LLE = p3.Estep_part2(X,K,Mu,P,Var)
MuM,PM,VarM = p3.Mstep_part2(X,K,Mu,P,Var,postE)

print "================================================"
print "posterior after first E step: ", postE
print "LL after first E step: ", LLE
print "================================================"
print "Mu after first M step: ", MuM
print "P after first M step: ", PM
print "Var after first M step: ", VarM 
print "================================================"


######## Test mixGauss and fillMatrix functions ########

MuEM, PEM, VarEM, postEM, LLEM = p3.mixGauss_part2(X,K,MuM,PM,VarM)
full = p3.fillMatrix(X,K,MuEM,PEM,VarEM)
error = p3.rmse(full,Xc)

print "================================================"
print "Mu: ", MuEM
print "P: ", PEM
print "Var: ", VarEM
print "post: ", postEM
print "LL: ", LLEM
print "================================================"
print "completed matrix: ", full
print "ground truth: ", Xc
print "================================================"
print "rmse: ", error
print "================================================"


############# Validate your solutions  ################ 
# Validate your solutions by comparing them to those
# provided in 'matrixCompletionTest_Solutions.txt'

# Pro tip: 
# To run this script in the terminal and write it to a 
# file called "validation.txt", you can use the line:
#
# python matrixCompletionTest.py >> validation.txt
#
# (making sure there is not already a file in the same folder
# called validation.txt; if there is, it will append the output
# to this existing file)

