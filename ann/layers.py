from abc import abstractmethod, ABCMeta

import numpy as np

class LayerFrame(metaclass=ABCMeta):
    def __init__(self):
        self.childLayer = None

    def __call__(self, layer):
        layer.childLayer = self
        return self
    
    @abstractmethod
    def forwardProp(self, x):
        pass

    @abstractmethod
    def backProb(self, dy):
        pass

class LossFunctionFrame(metaclass=ABCMeta):
    @abstractmethod

    def forwardProp(self, y_hat, y):
        pass

    @abstractmethod
    def backProb(self):
        pass

class InputLayer(LayerFrame):
    def __init__(self, shape):
        super().__init__()
        self.input_shape = tuple(shape)
    
    def forwardProp(self, x):
        x_shape = np.shape(x)[1:] #배치 차원 제외
        if x_shape != self.input_shape:
            raise
        return x

    def backProb(self, dy):
        return dy

class BPLayer(LayerFrame):
    def __init__(self, input_shape, units):
        super().__init__()
        self._rec_x = None

        self.W = np.random.rand(input_shape, units)
        self.b = np.random.rand(units)

    def forwardProp(self, x):
        xW = np.matmul(x, self.W)
        h = xW + self.b

        self._rec_x = x
        return h

    def backProb(self, dy):
        dx = np.dot(dy, self.W.T)

        #<Affine Method>
        db = dy.sum(axis=0)
        xT = np.transpose(self._rec_x)
        dW = np.dot(xT, dy)
        return dx, dW, db

        #<Personal Method>
        '''
        보편적인 방법에 해당하는 『Affine Method』의 경우,
        배치 별 가중치 미분 값을 합하여 출력한다.
        『Personal Methd』의 경우 배치에 따른 가중치를 그대로 출력한다.
        '''
        db = dy
        dW = np.multiply(dy, self._rec_x)
        return dx, dW, db

class Sigmoid(LayerFrame):
    def __init__(self):
        super().__init__()
        self._rec_o = None

    def forwardProp(self, x):
        o = 1 / (1 + np.exp(-x))
        
        self._rec_o = o
        return o

    def backProb(self, dy):
        do = self._rec_o * (1 - self._rec_o)
        return dy * do

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