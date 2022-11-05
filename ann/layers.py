from abc import abstractmethod, ABCMeta

import numpy as np

if __name__ == '__main__':
    from weightInitializers import Inintializers
else:
    from ann.weightInitializers import Inintializers

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
    def backProp(self, dy):
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

    def backProp(self, dy):
        return dy

class BPLayer(LayerFrame):
    def __init__(self, input_shape, units, weight_init=Inintializers.randomUniform):
        super().__init__()
        self._HAVE_WEIGHT = True
        self.weight_initializer = weight_init

        self.__rec_x = None

        self.W = self.weight_initializer(input_shape, units)
        self.b = np.full((units, ), 0.01, dtype=np.float64)

        self.dW = None
        self.db = None

    def forwardProp(self, x):
        xW = np.matmul(x, self.W)
        h = xW + self.b

        self.__rec_x = x.copy()
        return h

    def backProp(self, dy):
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

class FALayer(LayerFrame):
    def __init__(self, input_shape, units, weight_init=Inintializers.randomUniform):
        super().__init__()
        self._HAVE_WEIGHT = True
        self.weight_initializer = weight_init

        self.__rec_x = None

        #Task (1), (3): 𝐵는 균일 분포 [−0.5, 0.5]에서 추출된다(Methods Summary 참조).
        self.B = np.random.uniform(-0.5, 0.5, size=(units, input_shape))

        self.W = self.weight_initializer(input_shape, units)
        self.b = np.full((units, ), 0.01, dtype=np.float64)

        self.dW = None
        self.db = None

    def forwardProp(self, x):
        xW = np.matmul(x, self.W)
        h = xW + self.b

        self.__rec_x = x.copy()
        return h

    def backProp(self, dy):
        dx = np.dot(dy, self.B)

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

    def backProp(self, dy):
        do = self.__rec_o * (1 - self.__rec_o)
        return dy * do

class Tanh(LayerFrame):
    def __init__(self):
        super().__init__()
        self.__rec_o = None

    def forwardProp(self, x):
        o = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

        self.__rec_o = o
        return o

    def backProp(self, dy):
        return dy * (1 - self.__rec_o ** 2) 

class ReLU(LayerFrame):
    def __init__(self):
        super().__init__()

        self.__rec_x = None

    def forwardProp(self, x):
        self.__rec_x = x.copy()
        return np.maximum(0., x)

    def backProp(self, dy):
        do = (self.__rec_x > 0.) * 1.
        return dy * do

class LeakyReLU(LayerFrame):
    def __init__(self, alpha=0.3):
        super().__init__()
        self.alpha = alpha

        self.__rec_x = None

    def forwardProp(self, x):
        self.__rec_x = x.copy()
        return np.maximum(self.alpha * x, x)
    
    def backProp(self, dy):
        do = (self.__rec_x <= 0.) * 1.
        do = do * (self.alpha - 1.) + 1.
        return dy * do

class Softmax(LayerFrame):
    def __init__(self):
        super().__init__()
        self.__rec_softmax = None

    def forwardProp(self, x):
        max_ = np.max(x)
        exp_i = np.exp(x - max_)
        exp_sum = np.sum(exp_i, axis=1, keepdims=True)
        softmax = exp_i / exp_sum
        
        self.__rec_softmax = softmax
        return softmax

    def backProp(self, dy):
        batch_size, class_size = self.__rec_softmax.shape

        dy = np.expand_dims(dy, axis=1)
        soft = np.expand_dims(self.__rec_softmax, axis=1)
        softT = np.expand_dims(self.__rec_softmax, axis=2)

        I = np.eye(class_size)
        I = np.expand_dims(I, axis=0)
        I = np.repeat(I, batch_size, axis=0)

        dSoftmax = soft * (I - softT)
        dSoftmax = np.matmul(dy, dSoftmax)
        dSoftmax = np.squeeze(dSoftmax, axis=1)
        return dSoftmax
