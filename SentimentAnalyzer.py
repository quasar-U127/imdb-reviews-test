import tensorflow_datasets as tfds
import tensorflow as tf
from utils import Vocablury
import utils
import datetime
import os
import numpy as np


class SentimentAnalyzer(object):
    def __init__(self,
                 #  embedding_dim: int = 100,
                 rnn_unit: int = 128):
        super().__init__()
        self.vocab = Vocablury(
            load_prefix="sherlock")
        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(
                input_dim=len(self.vocab),
                output_dim=rnn_unit,
                mask_zero=True
                # output_dim=embedding_dim
            ),
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(units=rnn_unit, return_sequences=True)
            ),
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(units=rnn_unit)
            ),
            tf.keras.layers.Dense(units=rnn_unit, activation="relu"),
            tf.keras.layers.Dropout(rate=0.5),
            tf.keras.layers.Dense(units=1)
        ])

    def prepare_dataset(self):
        builder = tfds.builder("imdb_reviews")

        builder.download_and_prepare()
        datasets = builder.as_dataset()

        self.train: tf.data.Dataset = datasets["train"]
        self.test: tf.data.Dataset = datasets["test"]

    def prepare_for_training(self,
                             training_size: int = 80000,
                             batch_size: int = 64,
                             buffer_size: int = 10000
                             ):
        # convert to indexes
        # for x in self.train.take(1):
        #     print(x)
        self.training = self.train.map(utils.sentence_spliter_map_fn)
        self.training = self.training.flat_map(
            lambda text_list, label_list: tf.data.Dataset.zip(
                (
                    tf.data.Dataset.from_tensor_slices(text_list),
                    tf.data.Dataset.from_tensor_slices(label_list)
                )
            )
        )
        self.training = self.training.shuffle(buffer_size=buffer_size)

        for x, y in self.training.take(20):
            print(x)
            print(y)
            print()
        self.training = self.training.map(self.vocab.encode_map_fn)

        # size = self.training.reduce(np.int64(0), lambda x, _: x+1).numpy()
        # print(size, flush=True)
        # shuffle the dataset

        # split training,validation
        self.validation = self.training.skip(training_size)
        self.training = self.training.take(training_size)

        # # batching
        # # self.training = self.__batching__(self.training, batch_size=batch_size)
        # # self.validation = self.__batching__(
        # #     self.validation, batch_size=batch_size)

        self.training = self.training.padded_batch(
            batch_size=batch_size, drop_remainder=True)
        self.validation = self.validation.padded_batch(
            batch_size=batch_size, drop_remainder=True)

    def __call__(self, text):
        encoded_text = self.vocab.encode(tf.constant(text))
        return self.model(tf.expand_dims(encoded_text, 0))

    def train_model(self, epochs: int = 10):
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

        bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        self.model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                           loss=bce,
                           metrics=["accuracy"])
        print(self.model.summary())

        # Directory where the checkpoints will be saved
        checkpoint_dir = "./model/training_checkpoints"
        # Name of the checkpoint files
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_prefix,
            save_weights_only=True)

        history = self.model.fit(self.training,
                                 epochs=epochs,
                                 callbacks=[
                                     checkpoint_callback, tensorboard_callback],
                                 validation_data=self.validation)

    def load(self,  path_to_model: str = "./model/training_checkpoints"):
        self.model.load_weights("./model/training_checkpoints/ckpt_10")
        # self.model.load_weights(tf.train.latest_checkpoint(path_to_model))
