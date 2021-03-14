# biblioteker
import os
import shutil

"""
bruker tensorflow nightly:

pip3 install -q tensorflow_text_nightly --user
pip3 install -q tf-nightly --user

"""
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab


# laster inn og optimaliserer datasettene (se mer detaljert forklaring i selve modellen)
AUTOTUNE = tf.data.AUTOTUNE

batch_size = 32
seed = 1337

raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
    "./data/train",
    batch_size=batch_size,
    seed=seed
)

class_names = raw_train_ds.class_names
train_ds = raw_train_ds.cache().prefetch(buffer_size=AUTOTUNE)

# lag et datasett med bare tekst
bert_dataset = raw_train_ds.map(lambda x, y: x)


# innstillinger for tokenizer
# se mer detaljert forklaring i selve modellen
bert_tokenizer_params = dict(lower_case=True)
reserved_tokens = ["[PAD]", "[UNK]", "[START]", "[END]"]

bert_vocab_args = dict(
    vocab_size=8000 * 7,
    reserved_tokens=reserved_tokens,
    bert_tokenizer_params=bert_tokenizer_params,
    learn_params={},
)

# lager vokabularet baser på datasettet
vocab = bert_vocab.bert_vocab_from_dataset(
    bert_dataset,
    **bert_vocab_args
)
# lagrer vokabularet til en fil
os.remove("vocab.txt")
with open(f"vocab.txt", 'wb') as f:
    for token in vocab:
        f.write(token.encode() + b'\n')
