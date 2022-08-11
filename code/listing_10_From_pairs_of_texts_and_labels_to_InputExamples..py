def convert_text_to_examples(texts, labels):
    InputExamples = []
    for text, label in zip(texts, labels):
      InputExamples.append(
        InputExample(text=text, label=label)
      )
    return InputExamples
