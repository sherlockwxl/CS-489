from random import randint

import numpy


def lasso(x,y,l):
    n = len(x)
    d = len(x[0])
    w = numpy.zeros(d)
    lastw = w.copy()
    squaresum = numpy.sum(x**2,axis=0)

    while True:
        xw =x.copy()
        for i in range(0,d):
            xw[:,i]=x[:,i]*w[i]
        for j in range(0,d):
            a = squaresum[j]/2
            xw_new=numpy.delete(xw,j,1)
            xw_sum=numpy.sum(xw_new, 1)
            v=xw_sum-y
            b=numpy.sum(v*x[:,j])
            w[j] = numpy.sign((-b/(2*a)))*max(0,abs(-b/(2*a))-l/a)
            xw[:,j]=x[:,j]*w[j]
        if numpy.mean(numpy.abs(lastw - w)) < 0.001:
            return w
        else:
            lastw =w.copy()


def cross_validation(x,y,tx,ty):
    trainingsetx    = numpy.loadtxt(x, delimiter=",")
    trainingsety    = numpy.loadtxt(y, delimiter=",")
    testsetx        = numpy.loadtxt(tx, delimiter=",")
    testsety        = numpy.loadtxt(ty, delimiter=",")
    addtrainingsetx= numpy.random.standard_normal(size=(len(trainingsetx),1000))
    addtestsetx= numpy.random.standard_normal(size=(len(testsetx),1000))
    trainingsetx = numpy.hstack((trainingsetx,addtrainingsetx))
    testsetx = numpy.hstack((testsetx,addtestsetx))
    random = randint(0, trainingsetx.shape[0])
    #trainingsetx[random]*=10**6
    #trainingsety[random]*=10**3
    for l in range(0, 110, 10):
        sumtrain =0
        sumvalidation=0
        sumtest=0
        countnonzero=0
        for i in range(0,300,30):
            validationx  = trainingsetx[i:i+30]
            trainx   = numpy.vstack((trainingsetx[:i], trainingsetx[i+30:] ))
            validationy  = trainingsety[i:i+30]
            trainy   = numpy.hstack((trainingsety[:i], trainingsety[i+30:] ))
            w = lasso(trainx, trainy,l)
            nonzero = numpy.count_nonzero(w[13:])/(w.shape[0]-13)
            errortrain      = 1/len(trainx)*numpy.square(numpy.linalg.norm(numpy.dot(trainx,w)-trainy))
            errorvalidation = 1/len(validationx)*numpy.square(numpy.linalg.norm(numpy.dot(validationx,w)-validationy))
            errortest       = 1/len(testsetx)*numpy.square(numpy.linalg.norm(numpy.dot(testsetx,w)-testsety))
            sumtrain += errortrain
            sumvalidation += errorvalidation
            sumtest += errortest
            countnonzero+=nonzero
        print("lambda: ",l, " error of training set: ", sumtrain/10 , " error of validation set: ", sumvalidation/10,
              " error of test set: ", sumtest/10, " percentage of nonzero: ", countnonzero/10)

#ridge("housing_X_test.csv", "housing_y_test.csv", 0)
cross_validation("housing_X_train.csv", "housing_y_train.csv","housing_X_test.csv", "housing_y_test.csv")