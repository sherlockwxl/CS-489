import numpy
import csv
from numpy import *

def lasso(x,y,l):
    xsquare = numpy.sum(numpy.square(x),axis=0)
    w = numpy.zeros(x.shape[1])
    xw = numpy.zeros((x.shape))

    xwtemp = y.copy()
    xwtemp = xwtemp - 2*y
    w_backup = w.copy()

    while True:
        for j in range(0,x.shape[1]):
            a = 1/2 * xsquare[j]
            b = numpy.sum(x[:,j]*(xwtemp-xw[:,j]))
            z = max(0,abs(-b/(2*a))-l/a)
            if numpy.sign(b) > 0 :
                z = -z
            w[j] = z
            xwtemp = xwtemp - xw[:,j] + x[:,j]*z
            xw[:,j]=x[:,j]*z
            #print(numpy.mean(numpy.abs(w_backup - w)))
        if (numpy.linalg.norm(w_backup - w)) < 0.001:
            return w
        else:
            w_backup = w.copy()


def cross(x,y):
    trainx = numpy.loadtxt(x,delimiter=",")
    trainy = numpy.loadtxt(y,delimiter=",")

    #add test set
    testx = numpy.loadtxt("housing_X_test.csv",delimiter=",")
    testy = numpy.loadtxt("housing_Y_test.csv",delimiter=",")

    #norm for test
    #trainx = trainx/numpy.linalg.norm(trainx)
    #trainy = trainy / numpy.linalg.norm(trainy)
    #testx = testx/numpy.linalg.norm(testx)
    #testy = testy/numpy.linalg.norm(testy)
    # temp fix

    ilist =[]
    #test set done
    for i in range(0,11):  #i is for lamda
        #print(len(trainx[0]))
        #print(len(trainy))
        eachlength = int(len(trainy)/10) #eachlength is the length of each 1/10 train test
        errortrainingset = 0
        errorvalidtest = 0
        errortestset = 0
        nonzerocount = 0
        for startpoint in range(0,10):#used for 10 - fold test

            w = numpy.zeros((1,len(trainx[0])),dtype=float64)
            testsetstart = startpoint * eachlength
            testsetend = (startpoint+1) * eachlength
            if testsetend >= len(trainy):
                testsetend = len(trainy)

            #get sliced train set x
            testsetx = trainx[testsetstart:testsetend:1]
            trainsetxp1 = trainx[0:testsetstart:1]
            trainsetxp2 = trainx[testsetend:len(trainy):1]
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
                trainsety = numpy.hstack((trainsetyp1,trainsetyp2))


            w = lasso(trainsetx,trainsety,i*10)
            nonzerocount += numpy.count_nonzero(w)/(w.shape[0])


            #w = lasso2(trainsetx,trainsety,i*10)
            errorvalidtest += numpy.linalg.norm(numpy.dot(testsetx,w)-testsety,2)**2/len(testsety)
            errortrainingset += numpy.linalg.norm(numpy.dot(trainsetx,w)-trainsety,2)**2/len(trainsety)


        w = lasso(trainx, trainy, i*10)
        errortestset =  numpy.linalg.norm(numpy.dot(testx,w)-testy,2)**2/len(testy)


                #print(errortrainingset)
        print ("on $\lambda$ = %d  ,  training set error: %f, valid set mean error %f, test set error: %f, percentage of nonzeros in w %f \\\\"%(i*10,  errortrainingset/10,errorvalidtest/10, errortestset, nonzerocount/10))
        ilist.append((i,errorvalidtest,errortrainingset,errortestset,errorvalidtest+errortrainingset+errortestset))
    ilist.sort(key=lambda tup: tup[1])
    #print(ilist)
    ilist.sort(key=lambda tup: tup[2])
    #print(ilist)
    ilist.sort(key=lambda tup: tup[3])
    #print(ilist)





def main():

    import timeit

    start = timeit.default_timer()
    #filexn = input('Please enter test set for x: ')
    #fileyn = input('Please enter test set for y: ')
    filex = open("housing_x_train.csv")
    filey = open("housing_y_train.csv")
    cross(filex,filey)

    stop = timeit.default_timer()

    print(stop - start)


main()