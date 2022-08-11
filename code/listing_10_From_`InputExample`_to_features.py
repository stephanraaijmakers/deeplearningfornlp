def convert_single_example(tokenizer, example, max_seq_length=256): <1>
    tokens_a = tokenizer.tokenize(example.text) <2>
    if len(tokens_a) > max_seq_length - 2: <3>
        tokens_a = tokens_a[0 : (max_seq_length - 2)]
    tokens = []
    segment_ids = []
    tokens.append("[CLS]") <3>
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens) <4>

    input_mask = [1] * len(input_ids) <5>

    while len(input_ids) < max_seq_length: <6>
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    return input_ids, input_mask, segment_ids, example.label <7>
