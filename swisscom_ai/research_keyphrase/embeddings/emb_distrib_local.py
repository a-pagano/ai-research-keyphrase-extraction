# Copyright (c) 2017-present, Swisscom (Schweiz) AG.
# All rights reserved.
#
#Authors: Kamil Bennani-Smires, Yann Savary

import os
import pty
import subprocess
from six import with_metaclass
import sent2vec


import numpy as np

from swisscom_ai.research_keyphrase.embeddings.emb_distrib_interface import EmbeddingDistributor, Singleton


class EmbeddingDistributorLocal(with_metaclass(Singleton, EmbeddingDistributor)):
    """
    Concrete class of @EmbeddingDistributor using a local installation of sent2vec
    https://github.com/epfml/sent2vec

    It works by creating a subprocess in which text are fed through stdin , and embeddings are read from the stdout
    This is temporary, we are waiting for the new version of the FastText Python Wrapper for a cleaner implementation.
    """

    def __init__(self, fasttext_model):
        self.model = sent2vec.Sent2vecModel()
        self.model.load_model(fasttext_model)

    def get_tokenized_sents_embeddings(self, sents):
        """
        @see EmbeddingDistributor
        """
        for sent in sents:
            if '\n' in sent:
                raise RuntimeError('New line is not allowed inside a sentence')
        return self.model.embed_sentences(sents)
