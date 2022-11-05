from tqdm import tqdm

class Model:
    def __init__(self, inputs, outputs):
        self.input_layer = inputs
        self.output_layer = outputs

        self.lossFunction = None
        self.optimizer = None
    
    def compile(self, lossFunction, optimizer):
        self.lossFunction = lossFunction
        self.optimizer = optimizer

    def predict(self, x):
        flow_layer = self.input_layer
        flow_data = x
        while flow_layer is not None:
            flow_data = flow_layer.forwardProp(x=flow_data)
            flow_layer = flow_layer.childLayer
        return flow_data

    def update_on_batch(self, dLoss):
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

    def train(self, x, y):
        batch_size = len(x)
        dataset_iter = tqdm(zip(x, y), total=batch_size)

        losses = []
        for x, y in dataset_iter:
            y_hat = self.predict(x=x)
            loss = self.lossFunction.forwardProp(y_hat=y_hat, y=y)

            dLoss = self.lossFunction.backProp()
            self.update_on_batch(dLoss)

            dataset_iter.set_description(f'Loss: {loss:.5f}')
            losses.append(loss)
        return losses

    def inference(self, x, y):
        batch_size = len(x)
        dataset_iter = tqdm(zip(x, y), total=batch_size)
        
        losses = []
        for x, y in dataset_iter:
            y_hat = self.predict(x=x)
            loss = self.lossFunction.forwardProp(y_hat=y_hat, y=y)

            dataset_iter.set_description(f'Loss: {loss:.5f}')
            losses.append(loss)
        return losses
