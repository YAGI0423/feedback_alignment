class Model:
    def __init__(self, inputs, outputs):
        self.input_layer = inputs
        self.output_layer = outputs

        self.lossFunction = None
        self.optimizer = None
    
    def compile(self, lossFunction, optimizer):
        '''
        regist loss function & optimizer
        '''
        self.lossFunction = lossFunction
        self.optimizer = optimizer

    def predict(self, x):
        '''
        입력 데이터 x에 대한 네트워크의 출력 반환
        '''
        flow_layer = self.input_layer
        flow_data = x
        while flow_layer is not None:
            flow_data = flow_layer.forwardProp(x=flow_data)
            flow_layer = flow_layer.childLayer
        return flow_data

    def update_on_batch(self, dLoss):
        '''
        하나의 배치에 대한 ΔLoss를 입력받아 네트워크의 가중치 및 편향 갱신
        '''
        flow_layer = self.output_layer
        d_flow_data = dLoss
        while flow_layer is not None:
            d_flow_data = flow_layer.backProp(dy=d_flow_data)
            if flow_layer.have_weight():
                flow_layer.W = self.optimizer.optimize( #Weight update
                    parameter=flow_layer.W,
                    gradient=flow_layer.dW
                )
                flow_layer.b = self.optimizer.optimize( #bias update
                    parameter=flow_layer.b,
                    gradient=flow_layer.db
                )
            flow_layer = flow_layer.parentLayer