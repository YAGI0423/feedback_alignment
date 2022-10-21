from abc import abstractmethod, ABCMeta

import numpy as np

class LayerFrame(metaclass=ABCMeta):
    def __init__(self):
        self._HAVE_WEIGHT = False

        self.parentLayer = None
        self.childLayer = None

    def __call__(self, layer):
        layer.childLayer = self
        self.parentLayer = layer
        return self

    def have_weight(self):
        return self._HAVE_WEIGHT
    
    @abstractmethod
    def forwardProp(self, x):
        pass

    @abstractmethod
    def backProb(self, dy):
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
        self._HAVE_WEIGHT = True

        self.__rec_x = None

        self.W = np.random.rand(input_shape, units)
        self.b = np.random.rand(units)

        self.dW = None
        self.db = None

    def forwardProp(self, x):
        xW = np.matmul(x, self.W)
        h = xW + self.b

        self.__rec_x = x
        return h

    def backProb(self, dy):
        dx = np.dot(dy, self.W.T)

        #<Affine Method>
        self.db = dy.sum(axis=0)
        
        xT = np.transpose(self.__rec_x)
        self.dW = np.dot(xT, dy)
        return dx

        #<Personal Method>
        '''
        보편적인 방법에 해당하는 『Affine Method』의 경우,
        배치 별 가중치 미분 값을 합하여 출력한다.
        『Personal Methd』의 경우 배치에 따른 가중치를 그대로 출력한다.
        '''
        self.db = dy
        self.dW = np.multiply(dy, self.__rec_x)
        return dx

class Sigmoid(LayerFrame):
    def __init__(self):
        super().__init__()
        self.__rec_o = None

    def forwardProp(self, x):
        o = 1 / (1 + np.exp(-x))
        
        self.__rec_o = o
        return o

    def backProb(self, dy):
        do = self.__rec_o * (1 - self.__rec_o)
        return dy * do