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

def create_network(affine_type: str='BP', add_layer: bool=False):
    '''
    `Task (1) Linear function approximation`에서 사용한 모델 반환

    30-20-10 선형 네트워크가 선형 함수, 𝑇를 근사하도록 학습하였다.
    𝐵는 균일 분포 [−0.5, 0.5]에서 추출된다(=TaskInit).
    (Methods Summary 참조)
    '''
    if affine_type == 'BP': #Back Propagation
        AffineLayer = layers.BPLayer
    elif affine_type == 'FA': #Feedback Alignment
        AffineLayer = layers.FALayer
    else: #except
        raise  Exception('\n\n\nThe parameter `affine_type` must be "BP" or "FA" in string format.\n\n')

    inputs = layers.InputLayer(shape=(30, ))
    out = AffineLayer(input_shape=30, units=20, weight_init=Inintializers.Xavier)(inputs)
    out = layers.Tanh()(out)

    out = AffineLayer(input_shape=20, units=10, weight_init=Inintializers.TaskInit)(out)

    if add_layer: #30-20-10-10 Network
        out = layers.Tanh()(out)
        out = AffineLayer(input_shape=10, units=10, weight_init=Inintializers.TaskInit)(out)

    model = TaskModel(inputs=inputs, outputs=out)
    return model

def moving_average(x, window=10):
    '''
    x의 이동평균을 반환하는 함수
    '''
    x = list(x).copy()
    iter_num = len(x) - window - 1

    averages = []
    for _ in range(iter_num):
        mean = sum(x[:window]) / window
        averages.append(mean)
        del x[0]
    return averages

if __name__ == '__main__':
    EPOCH = 10
    BATCH_SIZE = 4
    LEARNING_RATE = 0.1

    dataset = loader.NonlinearFunctionApproximation(train_dataset_size=25000, input_shape=30, output_shape=10, is_normalize=True)

    networks = {
        'bp2_model': create_network(affine_type='BP', add_layer=False),
        'fa2_model': create_network(affine_type='FA', add_layer=False),
        'bp3_model': create_network(affine_type='BP', add_layer=True),
        'fa3_model': create_network(affine_type='FA', add_layer=True)
    }
    
    train_losses, test_losses = dict(), dict()
    for name, net in networks.items():
        net.compile(lossFunction=lossFunctions.SE(), optimizer=optimizers.SGD(learning_rate=LEARNING_RATE))
        train_loss, test_losses[name] = net.update_network(dataset=dataset, epoch=EPOCH, batch_size=BATCH_SIZE)

        train_losses[name] = moving_average(train_loss, window=500)

    historyVisualizer.visualize(
        title='Task (3) Nonlinear function approximation',
        path='./plot/images/task3_nonlinearFunction.png',
        train_losses=train_losses,
        test_losses=test_losses,
        epoch=EPOCH, tick_step=5,
        test_Ylim=(min(test_losses['fa3_model'])-0.001, max(test_losses['fa3_model'])+0.001)
    )