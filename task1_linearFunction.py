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
    out = AffineLayer(input_shape=30, units=20, weight_init=Inintializers.TaskInit)(inputs)
    out = AffineLayer(input_shape=20, units=10, weight_init=Inintializers.TaskInit)(out)
    model = TaskModel(inputs=inputs, outputs=out)
    return model

if __name__ == '__main__':
    EPOCH = 1000
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001

    dataset = loader.LinearFunctionApproximation(train_dataset_size=25000, input_shape=30, output_shape=10, is_normalize=True)

    bp_model = create_network(affine_type='BP')
    fa_model = create_network(affine_type='FA')

    bp_model.compile(lossFunction=lossFunctions.SE(), optimizer=optimizers.SGD(learning_rate=LEARNING_RATE))
    fa_model.compile(lossFunction=lossFunctions.SE(), optimizer=optimizers.SGD(learning_rate=LEARNING_RATE))

    bp_train_his, bp_test_his = bp_model.update_network(dataset=dataset, epoch=EPOCH, batch_size=BATCH_SIZE)
    fa_train_his, fa_test_his = fa_model.update_network(dataset=dataset, epoch=EPOCH, batch_size=BATCH_SIZE)


    historyVisualizer.visualize(
        title='Task (1) Linear function approximation',
        path='./plot/images/task1_linearFunction.png',
        train_losses={'BP': bp_train_his, 'FA': fa_train_his},
        test_losses={'BP': bp_test_his, 'FA': fa_test_his},
        epoch=EPOCH, tick_step=100
    )