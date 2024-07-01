import os
import joblib
import pandas as pd
import numpy as np
import warnings
from processing import preprocess_sentence

def load_data(dataset_path, sample_percent=100, shuffle=True, encoding = "ISO-8859-1",
              columns=["target", "ids", "date", "flag", "user", "text"]):
    df = pd.read_csv(dataset_path, encoding=encoding, names=columns)

    if 0 > sample_percent or sample_percent > 100:
        warnings.warn('Sample_percent value{} ignored. Sample_percent parameter must to be [0, 100].'.format(sample_percent), stacklevel=2)

    if 0 < sample_percent < 100:
        df = df.sample(frac=sample_percent)

    if np.logical_not(shuffle):
        warnings.warn('Shuffle=false, might cause unconvergence behavior', stacklevel=2)

    if shuffle:
        df.apply(np.random.shuffle, axis=0)

    # decode_map = {0: "NEGATIVE", 4: "POSITIVE"}
    df = df.loc[df['target'].isin([0, 4])]
    df['target'] = df['target'].replace(4, 1)
    df['text'] = df['text'].apply(preprocess_sentence)
    return df['text'], df['target']


def load_train_test(train_path="./data/training.1600000.processed.noemoticon.csv",
                    test_path="./data/testdata.manual.2009.06.14.csv",
                    cache_dir="./cache", cache_file="preprocessed_data.pkl"):

    if not os.path.exists(cache_dir):  
        os.makedirs(cache_dir)

    cache_data = None
    if cache_file is not None:
        try:
            with open(os.path.join(cache_dir, cache_file), "rb") as f:
                cache_data = joblib.load(f)
            print("Read preprocessed data from cache file:", cache_file)
        except:
            pass 

    if cache_data is None:
        print("Processing data")
        words_train, labels_train = load_data(train_path)
        words_test, labels_test = load_data(test_path)

        
        if cache_file is not None:
            cache_data = dict(words_train=words_train, words_test=words_test,
                              labels_train=labels_train, labels_test=labels_test)
            with open(os.path.join(cache_dir, cache_file), "wb") as f:
                joblib.dump(cache_data, f)

            print("Wrote preprocessed data to cache file:", cache_file)
    else:
        # Unpack data loaded from cache file
        words_train, words_test, labels_train, labels_test = (cache_data['words_train'],
                                                              cache_data['words_test'], cache_data['labels_train'],
                                                              cache_data['labels_test'])

    return words_train, words_test, labels_train, labels_test



