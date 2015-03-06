from __future__ import division
from numpy import array, zeros, dot, multiply, ceil, power, sqrt
import argparse, sys

from sklearn import svm

import matplotlib.cm as cm
import matplotlib.pyplot as plt



class Numbers: #from knn homework
    """
        Class to store MNIST data
        """
    
    def __init__(self, location):
        # You shouldn't have to modify this class, but you can if
        # you'd like.
        
        import cPickle, gzip
        
        # Load the dataset
        f = gzip.open(location, 'rb')
        train_set, valid_set, test_set = cPickle.load(f)
        
        self.train_x, self.train_y = train_set
        self.test_x, self.test_y = valid_set
        
        #print len(self.train_x)
        #print len(self.train_y)
        
        self.train_x, self.train_y = self.filter(self.train_y,True)
        self.test_x, self.test_y = self.filter(self.test_y,False)
        
        f.close()

    def filter(self,y_set,isTrain):
        filter_x = []
        filter_y = []
        for index,value in enumerate(y_set):
            if value==3 or value==8:
                if isTrain:
                    filter_x.append(self.train_x[index])
                else:
                    filter_x.append(self.test_x[index])
                filter_y.append(value)
        return filter_x,filter_y

kINSP = array([(1, 8, +1),
               (7, 2, -1),
               (6, -1, -1),
               (-5, 0, +1),
               (-5, 1, -1),
               (-5, 2, +1),
               (6, 3, +1),
               (6, 1, -1),
               (5, 2, -1)])

kSEP = array([(-2, 2, +1),    # 0 - A
              (0, 4, +1),     # 1 - B
              (2, 1, +1),     # 2 - C
              (-2, -3, -1),   # 3 - D
              (0, -1, -1),    # 4 - E
              (2, -3, -1),    # 5 - F
              ])


def weight_vector(x, y, alpha):
    """
    Given a vector of alphas, compute the primal weight vector.
    """
    
    w = zeros(len(x[0]))
   
    for index,vector in enumerate(x):
        w = w + multiply((alpha[index]*y[index]),vector)
    # TODO: IMPLEMENT THIS FUNCTION

    return w


def find_support(x, y, w, b, tolerance=0.001):
    """
    Given a primal support vector, return the indices for all of the support
    vectors
    """
    support = set()
    for index,vector in enumerate(x):
        #print vector
        #print index
        local = y[index]*(dot(w,vector)+b)
        #print "local: " + str(local)
        if 1-tolerance <= local <= 1+tolerance:
            
            support.add(index)
    #print x
    #print y
    #print w
    #print b
    #print "done"
#print support


    # TODO: IMPLEMENT THIS FUNCTION
    return support


def find_slack(x, y, w, b):
    """
    Given a primal support vector instance, return the indices for all of the
    slack vectors
    """

    slack = set()
    for index,vector in enumerate(x):
        #print vector
        #print index
        local = y[index]*(dot(w,vector)+b)
        #print "local: " + str(local)
        if local<1:
            
            slack.add(index)
    # TODO: IMPLEMENT THIS FUNCTION
    return slack

def computeSVM(svmType,data):
    if svmType == 'linear':
        clf = svm.SVC(kernel='linear')
    elif svmType == 'rbf':
        clf = svm.SVC()
    #pick the best value of c
    num_ex = int(ceil(0.6*len(data.train_y[:args.limit])))
    print num_ex
    #print ceil(0.6 * num_ex)
    bestC = 0
    minError = sys.maxint
    cv = []
    for i in range(-3,5):
        print "iteration: " +str(i+3)
        newC = power(10.0,i)
        clf.C = newC
        result = clf.fit(data.train_x[:num_ex], data.train_y[:num_ex])
        #print result
        error = 0
        for i,t in enumerate(data.train_x[num_ex:]):
            p = clf.predict(t)
            if p!=data.train_y[num_ex+i]:
                error+=1
    
        if error<minError:
            minError = error
            bestC = newC

        cv.append((newC,error,100*(error/len(data.train_y[num_ex:]))))
    
    print "minimum errors: " + str(minError)
    print "best C: " + str(bestC)
    
    clf.C = bestC
    result = clf.fit(data.train_x, data.train_y)
    error = 0
    for i,t in enumerate(data.test_x):
        p = clf.predict(t)
        if p!=data.test_y[i]:
            error+=1

    print "total errors: " +str(error)
    print len(data.test_y)
    print "percent error: " +str(100*(error/len(data.test_y)))

    if svmType == 'linear':
        return (cv,error,100*(error/len(data.test_y)),clf.support_vectors_)

    else:
        return (cv,error,100*(error/len(data.test_y)),None)

#My sci-kit learn SVM
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SVM Classifier Options')
    parser.add_argument('--k', type=int, default=3,
    help="Number of nearest points to use")
    parser.add_argument('--limit', type=int, default=-1,
    help="Restrict training to this many examples")

    args = parser.parse_args()

    data = Numbers("../data/mnist.pkl.gz")
    #print data.train_x[:args.limit], data.train_y[:args.limit]
    (cvLinear,t_errorLinear,p_errorLinear,support_vectorsLinear) = computeSVM('linear',data)
    #(cvRBF,t_errorRBF,p_errorRBF,support_vectorsRBF) = computeSVM('rbf',data)

    print "Linear Results"
    print cvLinear
    print t_errorLinear
    print p_errorLinear
    
    print "RBF Results"
    print cvRBF
    print t_errorRBF
    print p_errorRBF


    n = len(support_vectorsLinear[0])
    n_root = sqrt(n)
    plt.imshow(support_vectorsLinear[0].reshape((n_root, n_root)), cmap = cm.Greys_r)
    plt.show()
    plt.imshow(support_vectorsLinear[1].reshape((n_root, n_root)), cmap = cm.Greys_r)
    plt.show()
    plt.imshow(support_vectorsLinear[2].reshape((n_root, n_root)), cmap = cm.Greys_r)
    plt.show()

    #print support_vectorsLinear[0]
    #print support_vectorsLinear[1]
    #print support_vectorsLinear[2]



'''
