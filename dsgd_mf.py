import pyspark
import numpy as np
import sys
import random
import csv
from scipy import sparse

# Imports a sparse matrix csv into a scipy sparse matrix
def to_matrix(fileName):
    V = []
    R = []
    C = []
    csvfile = open(fileName, 'r')
    csvReader = csv.reader(csvfile)
    for row in csvReader:
        r = int(row[0])
        c = int(row[1])
        v = float(row[2])
        V.append(v)
        R.append(r - 1)
        C.append(c - 1)
    return sparse.csr_matrix((V, (R, C)))


def distributed_mf(V, factors, workers, maxIter):
    (m, n) = V.shape
    iteration = 0
    W = np.random.rand(m, factors)
    H = np.random.rand(factors, n)
    blockR = m / workers
    blockC = n / workers
    rowRanges = []
    colRanges = []
    for i in xrange(workers - 1):
        rowRanges.append((i * blockR, (i+1) * blockR))
        colRanges.append((i * blockC, (i+1) * blockC))
    rowRanges.append(((workers - 1) * blockR, m))
    colRanges.append(((workers - 1) * blockC, n))

    # Create nis njs. Those are lists of 
    nis = []
    njs = []
    for i in xrange(m):
        nis.append(len(sparse.find(V[i, :])[0]))
    for j in xrange(n):
        njs.append(len(sparse.find(V[:, j])[0]))

    numUpdates = 0
    updatedPerIter = len(sparse.find(V)[0])
    while(iteration < maxIter):
        for blockIter in xrange(workers):
            strata = []
            for blockNum in xrange(workers):
                (ra, rb) = rowRanges[blockNum]
                (ca, cb) = colRanges[(blockNum +blockIter) % workers]
                strata.append((V[ra:rb, ca:cb], W[ra:rb,:], H[:,ca:cb], \
                    numUpdates, workers, nis[ra:rb], njs[ca:cb]))
            
            results = sc.parallelize(strata, \
                workers).map(lambda x: SGD(x)).take(workers)
            for blockNum in xrange(workers):
                (ra, rb) = rowRanges[blockNum]
                (ca, cb) = colRanges[(blockNum +blockIter) % workers]
                (W[ra:rb,:], H[:,ca:cb]) = results[blockNum]
        iteration += 1
        # Code commented out for storing recon errors.
        # print >>f1, str(iteration) + "\t" + str(reconError(V,W,H))
        numUpdates += updatedPerIter
    # print reconError(V,W,H)
    np.savetxt(w_path, W, delimiter=",")
    np.savetxt(h_path, H, delimiter=",")

def reconError(V, W, H):
    err = 0
    (rowIndicies, colIndicies, nonZeroes) = sparse.find(V)
    for idx in xrange(len(nonZeroes)):
        i = rowIndicies[idx]
        j = colIndicies[idx]
        Vij = nonZeroes[idx]
        Wi = W[i, :]
        Hj = H[:, j]
        Lij = pow((Vij - np.dot(Wi,Hj)), 2)
        err += Lij
    return err / len(nonZeroes)


def SGD((V, W, H, sgdIter, numWorkers, nis, njs)):
    global lambda_value, beta_value
    (rowIndicies, colIndicies, nonZeroes) = sparse.find(V)
    innerIter = 0
    if(len(nonZeroes) <= 0):
        return (W, H, 0)
    while innerIter < len(nonZeroes):
        idx = random.randint(0, len(nonZeroes) - 1)
        i = rowIndicies[idx]
        j = colIndicies[idx]
        Vij = nonZeroes[idx]
        Wi = W[i, :]
        Hj = H[:, j]
        Ni = nis[i]
        Nj = njs[j]
        stepSize = (1000.0 + sgdIter + innerIter)**(-beta_value)
        sqrtLij = (Vij - np.dot(Wi,Hj))
        wLoss = -2.0 * sqrtLij * Hj + (2.0 * lambda_value * Wi.T) / \
            float(Ni)
        W[i, :] = Wi - stepSize * wLoss
        hLoss = -2.0 * sqrtLij * Wi.T + (2.0 * lambda_value * Hj) / \
            float(Nj)
        H[:, j] = Hj - stepSize * hLoss

        # Check loss
        #if innerIter % 1000 == 0:
        #   print reconError(V, W, H)
        
        innerIter += 1
    return (W, H)

# Setup spark local with n workers.
conf = pyspark.SparkConf().setAppName("dsgd").setMaster( \
    "local["+sys.argv[2]+"]")
sc = pyspark.SparkContext(conf=conf)

# Code to write down errors 
#f1=open('errors.txt', 'w+')

# Global constants
num_factors = int(sys.argv[1])
num_workers = int(sys.argv[2])
num_iterations = int(sys.argv[3])
beta_value = float(sys.argv[4])
lambda_value = float(sys.argv[5])
v_path = sys.argv[6]
w_path = sys.argv[7]
h_path = sys.argv[8]

distributed_mf(to_matrix(v_path), num_factors, num_workers, num_iterations)


