def tokenize_special_tokens(sentence, tokenizer, label_all_tokens=True):
    #tokenize input sentence to get input_ids (ids of the tokens of the input sentence)
    tokenized_input = tokenizer(sentence['tokens'], truncation=True, is_split_into_words=True)
    labels = []

    #loop over the labels of the input sentence
    for i, label in enumerate(sentence['ner_tags']):
        #get labels of the tokenized sentence
        word_ids = tokenized_input.word_ids(batch_index=i)

        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_input['labels'] = labels
    return tokenized_input

def compute_metrics():
    return None