def finetuneBertModel(trainT, valT):
    model=buildBertModel()
    (train_input_ids, train_input_masks, train_segment_ids,
    [CA]train_labels)=trainT
    (val_input_ids,val_input_masks,val_segment_ids,val_labels)=valT

    model.fit(
        [train_input_ids, train_input_masks, train_segment_ids],
        train_labels,
        epochs=10,
        batch_size=64
    )
