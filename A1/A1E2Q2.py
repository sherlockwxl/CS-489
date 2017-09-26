import numpy
import csv
from numpy import *
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


    #add test set

    with open("housing_X_test.csv") as csvfile3:
        testxreader = csv.reader(csvfile3)
        rownum3 = 0
        colnum3 = 0
        for row3 in testxreader:
            colnum3 = len(row3)
            rownum3 = rownum3 + 1
            #print("col3 num  %d" % colnum3)  # now we have the col num
            #print("row3 num  %d" % rownum3)  # now we have the row num

    with open("housing_X_test.csv") as csvfile3:
        testxreader = csv.reader(csvfile3)

        testx = numpy.zeros((rownum3, colnum3), dtype=float64)
        temp = 0
        for row3 in testxreader:
            testx[temp] = testx[temp] + array(row3, dtype=float64)
            temp = temp + 1

    # temp fix
    with open("housing_Y_test.csv") as csvfile4:
        testyreader = csv.reader(csvfile4)
        rownum4 = 0
        colnum4 = 0
        for row4 in testyreader:
            colnum4 = len(row4)
            rownum4 = rownum4 + 1
            # print("col2 num  %d" % colnum2)  # now we have the col num
            # print("row2 num  %d" % rownum2)  # now we have the row num

    with open("housing_Y_test.csv") as csvfile4:
        testyreader = csv.reader(csvfile4)

        testy = numpy.zeros((rownum4, 1), dtype=float64)
        temp = 0
        for row4 in testyreader:
            testy[temp] = testy[temp] + array(row4, dtype=float64)
            temp = temp + 1
    #norm for test
    trainx = trainx/numpy.linalg.norm(trainx)
    trainy = trainy / numpy.linalg.norm(trainy)
    testx = testx/numpy.linalg.norm(testx)
    testy = testy/numpy.linalg.norm(testy)
    # temp fix

    #Q2 modify
    tuple = numpy.random.randint(0,rownum2-1,1)
    print("will modify tuple %d" %(tuple))

    trainx1 = trainx
    trainy1 = trainy
    trainx1[tuple] = trainx1[tuple] * math.pow(10,6)
    trainy1[tuple] = trainy1[tuple] * math.pow(10,3)
    #print(trainx1[tuple])
    #print(trainy1[tuple])
    #test set done
    for q2i in range(0,3):
        ilist = []
        if q2i == 0:
            trainx = trainx
        elif q2i == 1:
            trainy = trainy
        else:
            trainx = trainx
            trainy = trainy
        for i in range(0,10):
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
                    trainsety = numpy.vstack((trainsetyp1,trainsetyp2))



                trainsetxtran = numpy.transpose(trainsetx)

                #print (trainsety.shape)
                left=numpy.dot(trainsetxtran,trainsetx) + (i*10)*1*numpy.identity(trainsetxtran.shape[0])
                right = numpy.dot(trainsetxtran,trainsety)

                w = numpy.linalg.solve(left, right)
                #print(numpy.linalg.norm(numpy.dot(trainx,w)-trainy,2))
                errorvalidtest += numpy.linalg.norm(numpy.dot(testsetx,w)-testsety,2)**2/len(testsety)
                errortrainingset += numpy.linalg.norm(numpy.dot(trainx,w)-trainy,2)**2/len(trainy)
                errortestset +=  numpy.linalg.norm(numpy.dot(testx,w)-testy,2)**2/len(testy)
                #print(errortrainingset)
            print ("on i %d , valid set mean error %f, training set error: %f, test set error: %f"%(i, errorvalidtest/10, errortrainingset/10, errortestset/10))
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
    filex = open("Q2x.csv")
    filey = open("Q2y.csv")
    regression(filex,filey)


main()