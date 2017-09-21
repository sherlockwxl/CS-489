import numpy
import csv
from numpy import *
from array import array
def regression(x,csvfile2):
    trainx = numpy.loadtxt(x,delimiter=",")
    #trainy = numpy.loadtxt(y,delimiter=",")
    #trainy = numpy.transpose(trainy)

    #issue with y
    with open("Q2y.csv") as csvfile2:
        baseyreader = csv.reader(csvfile2)
        rownum2 = 0
        colnum2 = 0
        for row2 in baseyreader:
            colnum2 = len(row2)
            rownum2 = rownum2 + 1
            #print("col2 num  %d" % colnum2)  # now we have the col num
            #print("row2 num  %d" % rownum2)  # now we have the row num


    with open("Q2y.csv") as csvfile2:
        baseyreader = csv.reader(csvfile2)

        trainy = numpy.zeros((rownum2, 1),dtype=float64)
        temp = 0
        for row2 in baseyreader:
            trainy[temp] = trainy[temp] + numpy.asarray(row2,dtype=float64)
            temp = temp + 1




# temp fix
    for i in range(0,10):
        #print(len(trainx[0]))
        #print(len(trainy))
        eachlength = int(len(trainy)/10) #eachlength is the length of each 1/10 train test
        errortrainingset = 0
        for startpoint in range(0,10):
            testsetstart = startpoint * eachlength
            testsetend = (startpoint+1) * eachlength
            if testsetend >= len(trainy):
                testsetend = len(trainy)

            #get sliced train set x
            testsetx = trainx[testsetstart:testsetend:1]
            trainsetxp1 = trainx[0:testsetstart:1]
            trainsetxp2 = trainx[testsetend:len(trainy):1]
            #print(trainy.shape)
            #print(trainsetxp1.shape)
            #print(trainsetxp2.shape)
            if len(trainsetxp1) == 0 :
                trainsetx = trainsetxp2
            elif len(trainsetxp2) == 0 :
                trainsetx = trainsetxp1
            else :
                trainsetx = numpy.vstack((trainsetxp1,trainsetxp2))

            #get sliced train set y
            testsety = trainy[testsetstart:testsetend:1]
            trainsetyp1 = trainy[0:testsetstart:1]
            trainsetyp2 = trainy[testsetend:len(trainy):1]
            if len(trainsetyp1) == 0 :
                trainsety = trainsetyp2
            elif len(trainsetxp2) == 0 :
                trainsety = trainsetyp1
            else :
                trainsety = numpy.vstack((trainsetyp1,trainsetyp2))



            trainsetxtran = numpy.transpose(trainsetx)

            print (trainsety.shape)
            left=numpy.dot(trainsetxtran,trainsetx) + (i+2)*10*numpy.identity(trainsetxtran.shape[0])
            right = numpy.dot(trainsetxtran,trainsety)

            w = numpy.linalg.solve(left, right)
            #print(numpy.linalg.norm(numpy.dot(trainx,w)-trainy,2))
            errortrainingset += numpy.linalg.norm(numpy.dot(testsetx,w)-testsety,2)**2/len(trainy)
            print(errortrainingset)
        print ("on i %d , trainning set mean error %f" %(i, errortrainingset/10))





def main():
    #filexn = input('Please enter test set for x: ')
    #fileyn = input('Please enter test set for y: ')
    filex = open("Q2x.csv")
    filey = open("Q2y.csv")
    regression(filex,filey)


main()