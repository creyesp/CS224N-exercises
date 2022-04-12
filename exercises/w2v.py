# tokenizar las secuencias
# obtener el skip-gram para cada secuencia
# samplear las observaciones random

import io
import re
import string
import tqdm

import numpy as np

import tensorflow as tf
import tensorflow.keras as keras

SEED = 42


def generate_training_data(sequences, window_size, num_ns, vocab_size, seed):
    """
    Generates skip-gram pairs with negative sampling for a list of sequences
    (int-encoded sentences) based on window size, number of negative samples
    and vocabulary size.

    :param sequences:
    :param window_size:
    :param num_ns:
    :param vocab_size:
    :param seed:
    :return:
    """
    # Elements of each training example are appended to these lists.
    targets, contexts, labels = [], [], []

    # Build the sampling table for `vocab_size` tokens.
    sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)

    # Iterate over all sequences (sentences) in the dataset.
    for sequence in tqdm.tqdm(sequences):

        # Generate positive skip-gram pairs for a sequence (sentence).
        positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
            sequence,
            vocabulary_size=vocab_size,
            sampling_table=sampling_table,
            window_size=window_size,
            negative_samples=0)

        # Iterate over each positive skip-gram pair to produce training examples
        # with a positive context word and negative samples.
        for target_word, context_word in positive_skip_grams:
            context_class = tf.expand_dims(
                tf.constant([context_word], dtype="int64"), 1)
            negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
                true_classes=context_class,
                num_true=1,
                num_sampled=num_ns,
                unique=True,
                range_max=vocab_size,
                seed=SEED,
                name="negative_sampling")

            # Build context and label vectors (for one target word)
            negative_sampling_candidates = tf.expand_dims(
                negative_sampling_candidates, 1)

            context = tf.concat([context_class, negative_sampling_candidates], 0)
            label = tf.constant([1] + [0]*num_ns, dtype="int64")

            # Append each element from the training example to global lists.
            targets.append(target_word)
            contexts.append(context)
            labels.append(label)

    return targets, contexts, labels


class Word2Vec(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, num_ns):
        super(Word2Vec, self).__init__()
        self.target_embedding = keras.layers.Embedding(
            vocab_size,
            embedding_dim,
            input_length=1,
            # embeddings_constraint=keras.constraints.NonNeg,
            name="w2v_embedding")
        self.context_embedding = keras.layers.Embedding(
            vocab_size,
            embedding_dim,
            input_length=num_ns+1,
            # embeddings_constraint=keras.constraints.NonNeg,
            name="ctx_embeding")
        self.dot = keras.layers.Dot(axes=(1,2))

    def call(self, pair):
        """
        context: (batch, context)
        target: (batch,)
        dots: (batch, context)

        :param pair:
        :return:
        """
        target, context = pair

        if len(target.shape) == 2:
            target = tf.squeeze(target, axis=1)
        word_emb = self.target_embedding(target)
        context_emb = self.context_embedding(context)
        # dots = tf.einsum('be,bce->bc', word_emb, context_emb)
        dots = self.dot([word_emb, context_emb])
        return dots


path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text_ds = tf.data.TextLineDataset(path_to_file).filter(lambda x: tf.cast(tf.strings.length(x), bool))


vocab_size = 4096
sequence_length = 10
num_ns = 4

vectorize_layer = keras.layers.experimental.preprocessing.TextVectorization(
    max_tokens=vocab_size,
    output_mode='int',
    output_sequence_length=sequence_length,
)
vectorize_layer.adapt(text_ds.batch(1024))
text_vector_ds = text_ds.batch(1024).prefetch(tf.data.AUTOTUNE).map(vectorize_layer).unbatch()
sequences = list(text_vector_ds.as_numpy_iterator())

targets, contexts, labels = generate_training_data(
    sequences=sequences,
    window_size=2,
    num_ns=num_ns,
    vocab_size=vocab_size,
    seed=SEED)

targets = np.array(targets)
contexts = np.array(contexts)[:, :, 0]
labels = np.array(labels)


BATCH_SIZE = 1024
BUFFER_SIZE = 10000
dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
dataset = dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)


embedding_dim = 128
word2vec = Word2Vec(vocab_size, embedding_dim, num_ns)
word2vec.compile(optimizer='adam',
                 loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                 metrics=['accuracy'])

word2vec((np.array([1, 2]), np.array([[3, 4, 5, 6, 7], [7, 8, 9, 0, 10]])))
word2vec.fit(dataset, epochs=20,)

weights = word2vec.get_layer('w2v_embedding').get_weights()[0]
vocab = vectorize_layer.get_vocabulary()