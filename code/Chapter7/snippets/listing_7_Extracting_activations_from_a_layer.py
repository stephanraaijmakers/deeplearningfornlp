
import keras.backend as K <1>

def getLayerActivation(model, inputData, layerName):
    activation = []
    MODE=0 <2>
    inp = model.layers[0].input

    for layer in model.layers: <3>

        if layer.name==layerName: <4>

            func = K.function([inp, K.learning_phase()], [layer.output]) <5>
            activation = [func([inputData,MODE])[0]] <6>
            break

    return activation <7>
