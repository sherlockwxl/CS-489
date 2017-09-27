import numpy
import csv
from numpy import *
def regression(x,y):
    trainx = numpy.loadtxt(x,delimiter=",")
    trainy = numpy.loadtxt(y,delimiter=",")

    trainy = numpy.transpose(trainy)


    testx = numpy.loadtxt("housing_X_test.csv", delimiter=",")
    testy = numpy.loadtxt("housing_Y_test.csv", delimiter=",")
    # temp fix

    #Q2 modify
    tuple = numpy.random.randint(0,trainx.shape[0])
    print("will modify tuple %d" %(tuple))

    trainx1 = trainx.copy()
    trainy1 = trainy.copy()
    trainx1[tuple] = trainx1[tuple] * 10**6
    trainy1[tuple] = trainy1[tuple] * 10**3
    #print(trainx1[tuple])
    #print(trainy1[tuple])
    #test set done
    trainx_backup = trainx.copy()
    trainy_backup = trainy.copy()
    for q2i in range(0,3):
        ilist = []
        if q2i == 0:
            trainx = trainx1.copy()
            trainy = trainy_backup.copy()
        elif q2i == 1:
            trainy = trainy1.copy()
            trainx = trainx_backup.copy()
        else:
            trainx = trainx1.copy()
            trainy = trainy1.copy()
        for i in range(0,11):
            #print(len(trainx[0]))
            #print(len(trainy))
            eachlength = int(len(trainy)/10) #eachlength is the length of each 1/10 train test
            errortrainingset = 0
            errorvalidtest = 0
            errortestset = 0
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
                    trainsety = numpy.hstack((trainsetyp1,trainsetyp2))



                trainsetxtran = numpy.transpose(trainsetx)

                left=numpy.dot(trainsetxtran,trainsetx) + (i*10)*numpy.identity(trainsetxtran.shape[0])
                right = numpy.dot(trainsetxtran,trainsety)

                w = numpy.linalg.solve(left, right)
                errorvalidtest += numpy.linalg.norm(numpy.dot(testsetx,w)-testsety)**2/len(testsety)
                errortrainingset += numpy.linalg.norm(numpy.dot(trainsetx,w)-trainsety)**2/len(trainsetx)
            left = numpy.dot(numpy.transpose(trainx), trainx) + (i * 10)* numpy.identity(numpy.transpose(trainx).shape[0])
            right = numpy.dot(numpy.transpose(trainx), trainy)

            w = numpy.linalg.solve(left, right)
            errortestset +=  numpy.linalg.norm(numpy.dot(testx,w)-testy)**2/len(testy)
                #print(errortrainingset)
            print ("on i %d , valid set mean error %f, training set error: %f, test set error: %f"%(i*10, errorvalidtest/10, errortrainingset/10, errortestset))
            ilist.append((i*10,errorvalidtest,errortrainingset,errortestset,errorvalidtest+errortrainingset+errortestset))
        if q2i == 0:
            print("only update trainning set x")
        elif q2i == 1:
            print("only update trainning set y")
        else:
            print("will update both trainning set x and y")
        ilist.sort(key=lambda tup: tup[1])
        #print((ilist))
        ilist.sort(key=lambda tup: tup[2])
        #print((ilist))
        ilist.sort(key=lambda tup: tup[3])
        #print((ilist))





def main():
    #filexn = input('Please enter test set for x: ')
    #fileyn = input('Please enter test set for y: ')
    filex = open("housing_x_train.csv")
    filey = open("housing_y_train.csv")
    regression(filex,filey)


main()