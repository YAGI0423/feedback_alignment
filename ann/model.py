import layer

if __name__ == '__main__':
    faLayer = layer.BPLayer(input_shape=2, units=1)
    sigmoid = layer.Sigmoid()
    loss_func = layer.MSE()

    print('W: ', faLayer.W, end='\n\n')
    print('b: ', faLayer.b, end='\n\n')

    x = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [[0], [1], [1], [1]]

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
