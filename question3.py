import numpy as np
from matplotlib import pyplot as plt


def LoadData():
    data = np.loadtxt('binarydigits.txt')
    N, D = data.shape

    return data, N, D


def Likelihoods(data, mu):
    N, D = data.shape
    K = len(mu)
    prob = np.zeros((N, K))
    
    for i in range(N):
        for k in range(K):
            prob[i,k] = np.prod((mu[k]**data[i])*((1-mu[k])**(1-data[i])))
    
    return prob

def Responsibilities(data, pi, mu):
    prob = Likelihoods(data, mu)
    
    #step 2
    # calculate the numerator of the resp.s
    prob = prob*pi
    
    #step 3
    # calcualte the denominator of the resp.s
    row_sums = prob.sum(axis=1)[:, np.newaxis]
    
    # step 4
    # calculate the resp.s
    try:
        prob = prob/row_sums
        return prob
    except ZeroDivisionError:
        print("Division by zero occured in reponsibility calculations!")

def LogLikelihood(data, pi, mu):
    N, D = data.shape
    K = len(pi)

    responsibilities = Responsibilities(data, pi, mu)

    logLike = 0
    for i in range(N):
        sumK = 0
        for k in range(K):
            try:
                temp1 = ((mu[k]**data[i])*((1-mu[k])**(1-data[i])))
                temp1 = np.log(temp1.clip(min=1e-50))
                
            except:
                print("Problem computing log(probability)")
            sumK += responsibilities[i, k]*(np.log(pi[k])+np.sum(temp1))
        logLike += sumK

    return logLike

def MStep(data, responsibilities):
    N, D = data.shape
    K = len(responsibilities[1])

    Nk = np.sum(responsibilities, axis=0)
    mus = np.zeros((K,D))

    for k in range(K):
        mus[k] = np.sum(responsibilities[:,k][:,np.newaxis]*data,axis=0)

        try:
                mus[k] = mus[k]/Nk[k]   
        except ZeroDivisionError:
                print("Division by zero occured in Mixture of Bernoulli Dist M-Step!")
                break  

    return Nk / N, mus

def FitModel(data, K, nIter):
    N, D = data.shape

    # Intialise random Bernoulli parameters
    pi = np.array([k/100 for k in range(1,K+1)])
    mu = np.random.uniform(size=(K,D))
    # pi = np.array([1/k for k in range(1, K+1)])
    # mu = np.random.uniform(size=(K,D))
    #logLike = LogLikelihood(data, pi, mu)
    #responsibilities = Responsibilities(data, pi, mu)

    logLikeVector = np.zeros(nIter)
    iterVector = np.arange(1, nIter+1)
    for cIter in range(nIter):
        # Expectation Step
        logLike = LogLikelihood(data, pi, mu)
        responsibilities = Responsibilities(data, pi, mu)

        # Maximisation Step
        pi, mu = MStep(data, responsibilities)
        # print('Pi: ', pi.shape)
        # print('Mu: ', mu)


        logLike = LogLikelihood(data, pi, mu)
        logLikeVector[cIter] = logLike
        print('Loglikelihood after {} iterations: {}'.format(cIter+1, logLike))

    return pi, mu, logLikeVector, iterVector

def PlotLoglikelihood(logLikeVector, iterVector):
    plt.plot(iterVector, logLikeVector, 'k')
    plt.xlabel('Iterations')
    plt.ylabel('Loglikelihood')
    plt.xticks(iterVector)
    plt.show()

def PlotImages(mu):
    print(mu.shape)
    for k in range(len(mu)):
        parameter = mu[k]

        plt.figure()
        plt.imshow(np.reshape(parameter, (8,8)), interpolation='None', cmap='gray')
        plt.axis('off')
        plt.show()
        


# --------------------Testing---------------------------
data, N, D = LoadData()
pi, mu, logLikeVector, iterVector = FitModel(data, K=3, nIter=20)
PlotLoglikelihood(logLikeVector, iterVector)
PlotImages(mu)