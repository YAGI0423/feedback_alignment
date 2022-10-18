from abc import abstractmethod, ABCMeta
import numpy as np

class Layer(metaclass=ABCMeta):
    @abstractmethod
    def forwardProp(self, x):
        pass

    @abstractmethod
    def backProb(self, dy):
        pass

class LossFunction(metaclass=ABCMeta):
    @abstractmethod
    def forwardProp(self, y_hat, y):
        pass

    @abstractmethod
    def backProb(self):
        pass

class BPLayer(Layer):
    def __init__(self, input_shape, units):
        self.W = np.random.rand(input_shape, units)
        self.b = np.random.rand(units)

        self.W = np.array(
            [
                [0.34995712],
                [0.260819],
            ]
        )
        self.b = np.array([0.32036945])

        self._rec_x = None

    def forwardProp(self, x):
        xW = np.dot(x, self.W)
        h = xW + self.b

        self._rec_x = x
        return h

    def backProb(self, dy):
        db = dy
        dx = np.dot(dy, self.W.T)
        xT = np.transpose(self._rec_x)
        dW = xT * dy
        return dx, dW, db

class Sigmoid(Layer):
    def __init__(self):
        self._rec_o = None

    def forwardProp(self, x):
        o = 1 / (1 + np.exp(-x))
        
        self._rec_o = o
        return o

    def backProb(self, dy):
        do = self._rec_o * (1 - self._rec_o)
        return dy * do

class MSE(LossFunction):
    def __init__(self):
        self._rec_err = None

    def forwardProp(self, y_hat, y):
        err = np.subtract(y_hat, y)
        loss = 0.5 * err**2

        self._rec_err = err
        return loss

    def backProb(self):
        return self._rec_err



if __name__ == '__main__':
    faLayer = BPLayer(input_shape=2, units=1)
    sigmoid = Sigmoid()
    loss_func = MSE()

    print('W: ', faLayer.W, end='\n\n')
    print('b: ', faLayer.b, end='\n\n')

    x = [[0, 0], [0, 1]]
    y = [[0], [1]]

    print('x: ', x, end='\n\n')
    print('y: ', y, end='\n\n')

    h = faLayer.forwardProp(x=x)
    print('h: ', h, end='\n\n')

    o = sigmoid.forwardProp(x=h)
    print('o: ', o, end='\n\n')

    loss = loss_func.forwardProp(y_hat=o, y=y)
    print('loss: ', loss, end='\n\n')

    print('=' * 50)

    do = loss_func.backProb()
    print('do: ', do, end='\n\n')

    dh = sigmoid.backProb(dy=do)
    print('dh: ', dh, end='\n\n')

    dx, dW, db = faLayer.backProb(dy=dh)
    print('dx: ', dx, end='\n\n')
    print('dW: ', dW, end='\n\n')
    print('db: ', db, end='\n\n')

    
    