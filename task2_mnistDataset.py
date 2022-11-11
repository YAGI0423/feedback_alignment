#<Network builder>
from ann import layers
from ann.weightInitializers import Inintializers
from taskModel import TaskModel

#<Network compile>
from ann import lossFunctions
from ann import optimizers

#<Dataset>
from datasets import loader

#<Visualizer>
from plot import historyVisualizer

def create_network(affine_type: str='BP'):
    '''
    `Task (2) MNIST dataset`에서 사용한 모델 반환

    표준 시그모이드 은닉과 출력 유닛(즉, 𝜎(𝑥) = 1 / (1+ exp (−𝑥)))의 784-1000-10 네트워크는
    0-9의 필기 숫자 이미지를 분류하도록 학습되었다.
    각 유닛은 조정 가능한 입력 편향(bias)이 있다.
    (Methods Summary 참조)
    '''
    if affine_type == 'BP': #Back Propagation
        AffineLayer = layers.BPLayer
    elif affine_type == 'FA': #Feedback Alignment
        AffineLayer = layers.FALayer
    else: #except
        raise  Exception('\n\n\nThe parameter `affine_type` must be "BP" or "FA" in string format.\n\n')

    inputs = layers.InputLayer(shape=(784, ))
    out = AffineLayer(input_shape=784, units=1000, weight_init=Inintializers.TaskInit)(inputs)
    out = layers.Sigmoid()(out)

    out = AffineLayer(input_shape=1000, units=10, weight_init=Inintializers.TaskInit)(out)
    out = layers.Sigmoid()(out)

    model = TaskModel(inputs=inputs, outputs=out)
    return model


if __name__ == '__main__':
    EPOCH = 20
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001

    dataset = loader.Mnist(is_normalize=True, is_one_hot=True)

    bp_model = create_network(affine_type='BP')
    fa_model = create_network(affine_type='FA')

    bp_model.compile(lossFunction=lossFunctions.SE(), optimizer=optimizers.SGD(learning_rate=LEARNING_RATE))
    fa_model.compile(lossFunction=lossFunctions.SE(), optimizer=optimizers.SGD(learning_rate=LEARNING_RATE))

    bp_train_his, bp_test_his = bp_model.update_network(dataset=dataset, epoch=EPOCH, batch_size=BATCH_SIZE)
    fa_train_his, fa_test_his = fa_model.update_network(dataset=dataset, epoch=EPOCH, batch_size=BATCH_SIZE)
    
    historyVisualizer.visualize(
        title='Task (2) MNIST dataset',
        path='./plot/images/task2_mnistDataset.png',
        train_losses={'BP': bp_train_his, 'FA': fa_train_his},
        test_losses={'BP': bp_test_his, 'FA': fa_test_his},
        epoch=EPOCH, tick_step=5
    )