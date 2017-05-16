from __future__ import division
import numpy as np
import random as rnd

def signum(x):
    if np.sign(x)>0:
        return 1
    else:
        return 0

#SVM


def find_bounds(alpha_i, alpha_j, y_i, y_j, C):
    if (y_i != y_j):
        return (max(0,alpha_j-alpha_i),min(C,C - alpha_i +alpha_j))
    else:
        return (max(0,alpha_j+alpha_i-C),min(C,alpha_j+alpha_i))

def bound_alpha(alpha_j,L,H):
    if alpha_j > H:
        return H
    elif alpha_j < L:
        return L
    else:
        return alpha_j




class Classificator():

    def __init__(self, C, gamma='auto',kernel_type ='rbf'):
        self.kernels = {
            'linear': self.kernel_linear,
            'quadratic': self.kernel_quadratic,
            'rbf': self.kernel_rbf
        }

        self.C = C
        self.gamma = gamma
        self.kernel_type = kernel_type


    def fit(self,X,y):

        X=X.as_matrix()
        y=y.as_matrix()
        n_samples, n_features = X.shape

        kernel = self.kernels[self.kernel_type]

        if self.gamma == 'auto':
            self.gamma = 1 / n_samples

        alpha = np.zeros((n_samples))
        count = 0
        eps=1



        while (eps>0.0001):

            j=0
            alpha_old = np.copy(alpha)
            for j in range(0,n_samples):

                i = j
                while(i == j):

                    i = rnd.randint(0,n_samples-1)


                alpha_i, alpha_j = alpha[i], alpha[j]
                x_i,x_j = X[i,:],X[j,:]
                y_i,y_j = y[i],y[j]

                alpha_idx = np.where(alpha > 0)[0]
                self.x_sup = X[alpha_idx, :]
                self.alpha_sup = alpha[alpha_idx,].reshape(-1,1)
                self.y_sup = y[alpha_idx,].reshape(-1,1)

                eta = -kernel(x_i,x_i)-kernel(x_j,x_j)+2*kernel(x_i,x_j)    #numbers


                (L,H) = find_bounds(alpha_i,alpha_j,y_i,y_j,self.C) #two numbers
                if(L==H):
                    continue
                print(L, H)


                self.b = y[j] - self.h_wob(self.x_sup, X[j, :], self.alpha_sup, self.y_sup)     # number



                error_i = signum (  (self.h_wob(self.x_sup, X[i, :], self.alpha_sup, self.y_sup) + self.b) - y_i   )  #number
                error_j = signum (  (self.h_wob(self.x_sup, X[j, :], self.alpha_sup, self.y_sup) + self.b) - y_j   )        #number


                alpha[j]= alpha_j - float((y_j * (error_i-error_j))/eta)

                print ("")


                print(alpha[j])

                alpha[j] = bound_alpha(alpha[j],L,H)


                print(alpha[j])

                alpha[i] = alpha_i + (y_i*y_j)*(alpha_j-alpha[j])



            count += 1
            eps = np.linalg.norm( alpha - alpha_old)



    def predict(self,X):
        X= X.as_matrix()
        result = []
        for j in range(0,X.shape[0]):
           result.append(signum((self.h_wob(self.x_sup, X[j, :], self.alpha_sup, self.y_sup) + self.b)))
        return result

    def h_wob(self,x1, x2, alpha_sup, y_sup):
        kernel = self.kernels[self.kernel_type]
        summa = 0
        for i in range(0, x1.shape[0]):
            summa += alpha_sup[i, :] * y_sup[i, :] * kernel(x1[i, :], x2)
        return summa
    def kernel_rbf(self,x1, x2):

        return np.exp(-(-2 * np.dot(x1, x2.T) + np.dot(x1, x1) + np.dot(x2, x2)) * self.gamma)

    def kernel_linear(self, x1, x2):
        return np.dot(x1, x2.T)

    def kernel_quadratic(self, x1, x2):
        return ((np.dot(x1, x2.T)+1) ** 2)


