from abc import abstractmethod, ABCMeta

import numpy as np

class LossFunctionFrame(metaclass=ABCMeta):
    @abstractmethod

    def forwardProp(self, y_hat, y):
        pass

    @abstractmethod
    def backProb(self):
        pass

class MSE(LossFunctionFrame):
    def __init__(self):
        self._rec_err = None

    def forwardProp(self, y_hat, y):
        err = np.subtract(y_hat, y)
        loss = np.square(err)
        loss = np.mean(loss)

        self._rec_err = err
        return loss

    def backProb(self):
        batch_by_class = np.prod(self._rec_err.shape) #batch size * class num
        dloss = (2 / batch_by_class) * self._rec_err
        
        #<Batch x class로 나누는 이유에 대해>
        '''
        손실함수 단계에서 미리 미분값을 배치 크기로 나누면,
        이후 Affine 계층의 ∂L/∂W 값을 matmul(Δy, Δx.T) 만으로 배치 크기를 고려하여 구할 수있다.
        해당 연산이 없는 경우, 동일한 미분값을 얻기위해서는 각 ΔW를 배치 크기로 나누어주어야 하는 번거로움이 있다. 
        '''
        return dloss