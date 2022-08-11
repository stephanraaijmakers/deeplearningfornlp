def convert_examples_to_features(tokenizer, examples, max_seq_length=256):
    input_ids, input_masks, segment_ids, labels = [], [], [], []
    for example in examples:
      input_id, input_mask, segment_id, label = convert_single_example(
      [CA]tokenizer, example, max_seq_length) <1>
      input_ids.append(input_id) <2>
      input_masks.append(input_mask)
      segment_ids.append(segment_id)
      labels.append(label)
    return (
        np.array(input_ids),
        np.array(input_masks),
        np.array(segment_ids),
        np.array(labels).reshape(-1, 1)
    ) <3>
