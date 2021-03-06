#!/usr/bin/env python
import sys

from subprocess import *
import resource
import time
import numpy as np
from numpy import linalg
from scipy import sparse
import csv

def LoadMatrix(csvfile):
	data = np.genfromtxt(csvfile, delimiter=',')
	return np.matrix(data)

def LoadSparseMatrix(csvfile):
        val = []
        row = []
        col = []
        select = []
        f = open(csvfile)
        reader = csv.reader(f)
        for line in reader:
                row.append( int(line[0])-1 )
                col.append( int(line[1])-1 )
                val.append( int(line[2]) )
                select.append( (int(line[0])-1, int(line[1])-1) )
        return sparse.csr_matrix( (val, (row, col)) ), select

def CalculateError(V, W, H, select):
        diff = V-W*H
        error = 0
        for row, col in select:
                error += diff[row, col]*diff[row, col]
        return error/len(select)



W = LoadMatrix(sys.argv[len(sys.argv)-2])
H = LoadMatrix(sys.argv[len(sys.argv)-1])
V, select = LoadSparseMatrix(sys.argv[len(sys.argv)-3])
error = CalculateError(V,W,H,select)
print "Reconstruction error:", error

