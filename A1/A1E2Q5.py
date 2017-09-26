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

    adddataset = numpy.random.uniform(-1, 1, size=(len(trainx), 1000))
    trainx = numpy.hstack((trainx, adddataset))
    adddataset2 = numpy.random.uniform(-1, 1, size=(len(testx), 1000))
    testx = numpy.hstack((testx, adddataset2))
    #norm for test
    trainx = trainx/numpy.linalg.norm(trainx)
    trainy = trainy / numpy.linalg.norm(trainy)
    testx = testx/numpy.linalg.norm(testx)
    testy = testy/numpy.linalg.norm(testy)
    # temp fix

    ilist =[]
    #test set done
    for i in range(0,10):  #i is for lamda
        #print(len(trainx[0]))
        #print(len(trainy))
        eachlength = int(len(trainy)/10) #eachlength is the length of each 1/10 train test
        errortrainingset = 0
        errorvalidtest = 0
        errortestset = 0
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
            #left=numpy.dot(trainsetxtran,trainsetx) + (i)*1*numpy.identity(trainsetxtran.shape[0])
            #right = numpy.dot(trainsetxtran,trainsety)

            #w = numpy.linalg.solve(left, right)
            #print(numpy.linalg.norm(numpy.dot(trainx,w)-trainy,2))
            j = 1
            while True:#now loop untill change of w with 10-3

                position = j%len(trainsetx[0])# optimial jth w
                a = 0
                result = numpy.zeros((trainsetx.shape[0], 1), dtype=float64)
                b = 0
                c = 0
                for p in range(0,trainsetx.shape[0]):
                    a += (trainx[p][position])**2
                    resulttemp = numpy.inner(w,trainsetx[p])-trainsety[p]-trainx[p][position]*w[0][position]
                    result[p] = resulttemp
                    #print(numpy.inner(w,trainsetx[p]))
                    #print(trainsety[p])
                    #print(trainx[p][position])

                for p in range(1, trainsetx.shape[0]):
                    b += trainx[p][position]*result[p]
                    c += result[p]*result[p]

                #now we could get z
                print(abs(-b/(2*a))-i*10/a)
                z = max(0,abs(-b/(2*a))-i*10/a)
                if (b/a) > 0:
                    z = -z
                if ( abs(w[0][position]-z) < 10**(-3)):
                    break
                else:
                    w[0][position] = z
                    j += 1
                # now calculate V




            w = numpy.transpose(w)
            errorvalidtest += numpy.linalg.norm(numpy.dot(testsetx,w)-testsety,2)**2/len(testsety)
            errortrainingset += numpy.linalg.norm(numpy.dot(trainx,w)-trainy,2)**2/len(trainy)
            errortestset +=  numpy.linalg.norm(numpy.dot(testx,w)-testy,2)**2/len(testy)
            #print(errortrainingset)
        print ("on i %d , valid set mean error %f, training set error: %f, test set error: %f"%(i, errorvalidtest/10, errortrainingset/10, errortestset/10))
        ilist.append((i,errorvalidtest,errortrainingset,errortestset,errorvalidtest+errortrainingset+errortestset))
    ilist.sort(key=lambda tup: tup[1])
    #print(ilist)
    ilist.sort(key=lambda tup: tup[2])
    #print(ilist)
    ilist.sort(key=lambda tup: tup[3])
    #print(ilist)





def main():
    #filexn = input('Please enter test set for x: ')
    #fileyn = input('Please enter test set for y: ')
    filex = open("Q2x.csv")
    filey = open("Q2y.csv")
    regression(filex,filey)


main()