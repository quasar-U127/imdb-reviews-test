from typing import List
import tensorflow as tf
import tensorflow_datasets as tfds
import nltk


def get_vocab(name: str) -> List[str]:
    with open(name) as vocab_source:
        vocab = []
        for word in vocab_source:
            vocab += [word[:-1]]
    vocab = sorted(set(vocab))
    return vocab


def get_vocab_generator(name: str) -> List[str]:
    with open(name) as vocab_source:
        # vocab = []
        for word in vocab_source:
            # vocab += [word[:-1]]
            yield word
    # vocab = sorted(set(vocab))
    # return vocab


def sentence_spliter(text, label):
    # print(text.numpy(), flush=True)
    sentences = nltk.sent_tokenize(text.numpy().decode("utf-8"))
    return sentences, [label]*len(sentences)
    # return sentences, label


def sentence_spliter_map_fn(sample):
    sentences, labels = tf.py_function(sentence_spliter,
                                       inp=[sample["text"],
                                            sample["label"]],
                                       Tout=(tf.string, tf.int64))
    sentences.set_shape([None])
    labels.set_shape([None])
    return sentences, labels


class Vocablury(object):
    def __init__(self, source: str = None, load_prefix: str = None):
        if source:
            self.text_encoder = tfds.features.text.SubwordTextEncoder.build_from_corpus(
                corpus_generator=get_vocab_generator(source), target_vocab_size=20000)
        else:
            self.text_encoder = tfds.features.text.SubwordTextEncoder.load_from_file(
                load_prefix)

    def __len__(self):
        return self.text_encoder.vocab_size

    def encode_label(self, text_tensor, label):
        return self.text_encoder.encode(text_tensor.numpy()), label

    def encode(self, text_tensor):
        return self.text_encoder.encode(text_tensor.numpy())

    def encode_map_fn(self, text, label):
        encoded_text, label = tf.py_function(self.encode_label,
                                             inp=[text, label],
                                             Tout=(tf.int64, tf.int64))
        encoded_text.set_shape([None])
        label.set_shape([])
        return encoded_text, label

    def decode(self, tensor):
        return self.text_encoder.decode(tensor.numpy())

    def save(self):
        self.text_encoder.save_to_file("sherlock")
