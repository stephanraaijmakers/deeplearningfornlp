
import keras.backend as K 

def getLayerActivation(model, inputData, layerName):
    activation = []
    MODE=0 
    inp = model.layers[0].input

    for layer in model.layers: 

        if layer.name==layerName: 

            func = K.function([inp, K.learning_phase()], [layer.output]) 
            activation = [func([inputData,MODE])[0]] 
            break

    return activation 
