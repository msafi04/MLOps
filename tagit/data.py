# tagit/data.py
# Data processing operations.

import itertools
import json
import re
from collections import Counter
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from skmultilearn.model_selection import IterativeStratification

import tensorflow
import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K

def filter_items(items: List, include: List = [], exclude: List = []) -> List:
    """Filter a list using inclusion and exclusion lists of items.
    Args:
        items (List): List of items to apply filters.
        include (List, optional): List of items to include. Defaults to [].
        exclude (List, optional): List of items to filter out. Defaults to [].
    Returns:
        Filtered list of items.
    Usage:
    ```python
    # Filter tags for each project
    df.tags = df.tags.apply(
        filter_items,
        include=list(tags_dict.keys()),
        exclude=config.EXCLDUE,
        )
    ```
    """
    # Filter
    filtered = [item for item in items if item in include and item not in exclude]

    return filtered


def prepare(
    df: pd.DataFrame, include: List = [], exclude: List = [], min_tag_freq: int = 30
) -> Tuple:
    """Prepare the raw data.
    Args:
        df (pd.DataFrame): Pandas DataFrame with data.
        include (List): list of tags to include.
        exclude (List): list of tags to exclude.
        min_tag_freq (int, optional): Minimum frequency of tags required. Defaults to 30.
    Returns:
        A cleaned dataframe and dictionary of tags and counts above the frequency threshold.
    """
    # Filter tags for each project
    df.tags = df.tags.apply(filter_items, include=include, exclude=exclude)
    tags = Counter(itertools.chain.from_iterable(df.tags.values))

    # Filter tags that have fewer than `min_tag_freq` occurrences
    tags_above_freq = Counter(tag for tag in tags.elements() if tags[tag] >= min_tag_freq)
    tags_below_freq = Counter(tag for tag in tags.elements() if tags[tag] < min_tag_freq)
    df.tags = df.tags.apply(filter_items, include=list(tags_above_freq.keys()))

    # Remove projects with no more remaining relevant tags
    df = df[df.tags.map(len) > 0]

    return df, tags_above_freq, tags_below_freq


class Stemmer(PorterStemmer):
    def stem(self, word):

        if self.mode == self.NLTK_EXTENSIONS and word in self.pool:  # pragma: no cover, nltk
            return self.pool[word]

        if self.mode != self.ORIGINAL_ALGORITHM and len(word) <= 2:  # pragma: no cover, nltk
            # With this line, strings of length 1 or 2 don't go through
            # the stemming process, although no mention is made of this
            # in the published algorithm.
            return word

        stem = self._step1a(word)
        stem = self._step1b(stem)
        stem = self._step1c(stem)
        stem = self._step2(stem)
        stem = self._step3(stem)
        stem = self._step4(stem)
        stem = self._step5a(stem)
        stem = self._step5b(stem)

        return stem


def preprocess(
    text: str,
    lower: bool = True,
    stem: bool = False,
    filters: str = r"[!\"'#$%&()*\+,-./:;<=>?@\\\[\]^_`{|}~]",
    stopwords: stopwords
) -> str:
    """Conditional preprocessing on text.
    Usage:
    ```python
    preprocess(text="Transfer learning with BERT!", lower=True, stem=True)
    ```
    <pre>
    'transfer learn bert'
    </pre>
    Args:
        text (str): String to preprocess.
        lower (bool, optional): Lower the text. Defaults to True.
        stem (bool, optional): Stem the text. Defaults to False.
        filters (str, optional): Filters to apply on text.
        stopwords (List, optional): List of words to filter out. Defaults to STOPWORDS.
    Returns:
        Preprocessed string.
    """
    # Lower
    if lower:
        text = text.lower()

    # Remove stopwords
    if len(stopwords):
        pattern = re.compile(r"\b(" + r"|".join(stopwords) + r")\b\s*")
        text = pattern.sub("", text)

    # Remove URLs
    text = re.sub(r"http\S+", "", text)
    #numbers
    text = re.sub("\d+", "", text)
    # Spacing and filters
    text = re.sub(r"([-;;.,!?<=>])", r" \1 ", text)
    text = re.sub(filters, r"", text)
    text = re.sub("[^A-Za-z0-9]+", " ", text)  # remove non alphanumeric chars
    text = re.sub(" +", " ", text)  # remove multiple spaces
    text = ' '.join([w for w in text.split() if len(w) > 1]) #remove single char
    text = text.strip()

    # Stemming
    if stem:
        stemmer = Stemmer()
        text = " ".join([stemmer.stem(word) for word in text.split(" ")])

    return text

def iterative_train_test_split(X: pd.Series, y: np.ndarray, train_size: float = 0.7) -> Tuple:
    """Custom iterative train test split which
    'maintains balanced representation with respect
    to order-th label combinations.'
    Args:
        X (pd.Series): Input features as a pandas Series object.
        y (np.ndarray): One-hot encoded labels.
        train_size (float, optional): Proportion of data for first split. Defaults to 0.7.
    Returns:
        Two stratified splits based on specified proportions.
    """
    stratifier = IterativeStratification(
        n_splits=2,
        order=1,
        sample_distribution_per_fold=[
            1.0 - train_size,
            train_size,
        ],
    )
    train_indices, test_indices = next(stratifier.split(X, y))
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    return X_train, X_test, y_train, y_test

def get_data_splits(df: pd.Series, train_size = 0.7) -> Tuple:
    """Custom function to split input data and encode multi-labels"""
    # Get data
    X = df['text'].to_numpy()
    y = df['tags']
    
    # Binarize y
    label_encoder = MultiLabelBinarizer()
    label_encoder.fit(y)
    y = label_encoder.transform(y)

    # Split
    X_train, X_, y_train, y_ = iterative_train_test_split(
        X, y, train_size = train_size)
    X_val, X_test, y_val, y_test = iterative_train_test_split(
        X_, y_, train_size = 0.5)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, label_encoder

def get_tokenizer(df: pd.series):
    oov_token = '<UNK>'
    pad_type = 'post'
    trunc_type = 'post'

    tokenizer = Tokenizer(char_level = False, oov_token = oov_token)
    tokenizer.fit_on_texts(df['text'].values)
    return tokenizer

def tokenize_text(texts, tokenizer, max_sequence_length):
    # Turns text into into padded sequences.
    text_sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(text_sequences, padding = pad_type, truncating = 'post', maxlen = max_sequence_length)

class CNNTextDataset(tf.keras.utils.Sequence):
    """Create `Tf Dataset` objects to use for
        efficiently feeding data into our models.
        Usage:
        ```python
        # Create dataset
        X, y = data
        dataset = CNNTextDataset(X=X, y=y, max_len=max_len, batch_size=batch_size, num_classes=num_classes,
                                tokenizer=tokenizer, shuffle=True/False)
        X: text input
        y: labels
        max_len: maximum length of text sequence
        batch_size: batch size to feed into NN
        num_classes: number of classes to predict
        tokenizer: tokenizer object from Tensorflow Tokenizer
        shuffle: True if train, False if evaluation/prediction
        
        ```
        """
    def __init__(self, X, y, max_len, batch_size, num_classes, 
                tokenizer, shuffle = True):
        self.X = X
        self.y = y
        self.max_len = max_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_classes = num_classes
        self.tokenizer = tokenizer
        self.list_IDs = np.arange(len(self.X))
        self.on_epoch_end()
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.X) / self.batch_size))

    def __str__(self):
        return f"<Dataset(N={len(self)})>"

    def __getitem__(self, index):
        #'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y
    
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X_ = np.empty((self.batch_size, self.max_len), dtype = int)
        y_ = np.empty((self.batch_size, self.num_classes), dtype = int)
        
        # Generate data
        
        #train data
        X_ = self.X[list_IDs_temp]

        # train class
        y_ = self.y[list_IDs_temp]
        #print(X_, y_)

        X = tokenize_text(X_, self.tokenizer, self.max_len)
        y = y_
        
        # Cast
        X = tf.cast(np.asarray(X).astype('int32'), dtype = tf.int32)
        y = tf.cast(np.asarray(y).astype('int32'), dtype = tf.int32)
        
        return X, y
        
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.X))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


