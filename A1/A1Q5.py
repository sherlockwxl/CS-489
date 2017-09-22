import csv
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from decimal import Decimal
import warnings

max_pass = 500

with open('spambase_X.csv') as csvfile:
    basexreader = csv.reader(csvfile)
    rownum = 0
    colnum = 0
    for row in basexreader:
        colnum = len(row)
        rownum = rownum + 1

    print ("col num  %d" %colnum) # now we have the col num
    print ("row num  %d" % rownum)  # now we have the row num

with open('spambase_X.csv') as csvfile:
    basexreader = csv.reader(csvfile)
    matrix_x = np.zeros((rownum,(colnum+1) * 2),dtype=complex_)
    temp = 0

    for row in basexreader:

        tempfirst = array(row,dtype=float_)
        tempfirst= np.append(tempfirst,[1])
        print("col num  %d" % len(tempfirst))
        tempsecond = np.negative(tempfirst)
        transformx = np.append(tempfirst, tempsecond)
        matrix_x[temp] = matrix_x[temp] + transformx
        temp = temp + 1

with open('spambase_y.csv') as csvfile2:
    baseyreader = csv.reader(csvfile2)
    rownum2 = 0
    colnum2 = 0
    for row2 in baseyreader:
        colnum2 = len(row2)
        rownum2 = rownum2 + 1
    print ("col2 num  %d" % colnum2)  # now we have the col num
    print ("row2 num  %d" % rownum2)  # now we have the row num

if rownum != rownum2:
    print ("row number does not match")


with open('spambase_y.csv') as csvfile2:
    baseyreader = csv.reader(csvfile2)

    matrix_y = np.zeros((rownum,1),dtype=complex_)
    temp = 0
    for row2 in baseyreader:
        matrix_y[temp] = matrix_y[temp] + array(row2,dtype=float_)
        temp = temp + 1
#print(matrix_y[0])


#now inilizte w and start train
w = zeros((colnum+1) * 2,dtype=complex_)
w = w + 1/(1+colnum)
b = 1/(1+colnum)
n = 0.0069315 #0.69315# the step size
#print(w)
#print(b)
lstx = []
lsty = []
for t in range(0,500):
    mistake = 0

    for i in range(0,rownum):
        #print(w)
        #print(b)
        if(matrix_y[i,0] * (np.inner(matrix_x[i],w) + b)) <= 0:
            #print("wrong")
            #print(np.multiply(n,(matrix_y[i,0]*matrix_x[i])))
            temp = np.exp(np.multiply(n,(matrix_y[i,0]*matrix_x[i])))
            w = np.multiply(w, temp)
            b = b * np.exp(matrix_y[i,0]*n)
            #print(b)
            #print(np.sum(w))
            s = b + np.sum(w)

            w = np.divide(w,s)
            b = b/s
        #if (matrix_y[i, 0] * (np.inner(matrix_x[i], w) + b)) <= 0:
            mistake = mistake + 1



    print ("passes: %d, mistake: %d" %(t, mistake))
    lstx.append(t)
    lsty.append(mistake)

plt.plot(lstx,lsty,'ro')
#plt.axis([0, 500, 400, 1000])
plt.show()

