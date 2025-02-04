
def run_evaluate(trainingData, testData, context=False):

    X_tr,Q_tr,y_tr,X_te,Q_te,y_te,model=create_model(
    [CA]trainingData,testData,context)

    print('Training')
    model.fit([X_tr, Q_tr], y_tr,
              batch_size=32,
              epochs=10,
              verbose=1,
              validation_split=0.1)

    print('Evaluation')
    loss, acc = model.evaluate([X_te,Q_te], y_te,
                               batch_size=32)
    print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))
