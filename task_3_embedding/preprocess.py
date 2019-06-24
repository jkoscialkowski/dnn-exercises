import itertools
import re

from collections import Counter

from torch.utils.data import Dataset

class CorpusPreprocessor:
    def __init__(self, corpus_path):
        self.corpus_path = corpus_path
        with open(corpus_path, 'r', encoding='utf8') as f:
            self.raw_corpus = f.readlines(15 * 1024 * 1024)[:100000]

    def transform_text(self):
        # Remove non-alphabetic characters
        temp_corpus = map(lambda x: re.sub('[^\w ]+', '', x), self.raw_corpus)

        # Remove numbers
        temp_corpus = map(lambda x: re.sub('[0-9]+', '', x), temp_corpus)

        # Remove unnecessary whitespaces
        temp_corpus = map(lambda x: re.sub(' {2,}', '', x), temp_corpus)

        # Lowercase for all letters
        temp_corpus = map(lambda x: x.lower(), temp_corpus)

        self.transformed_corpus = list(temp_corpus)

    def extract_vocab(self):
        # Execute transform_text if not done before
        if not hasattr(self, 'transformed_corpus'):
            self.transform_text()

        # Transform the corpus to a list of words
        words = list(
            itertools.chain(
                *map(lambda x: x.split(), self.transformed_corpus)
            )
        )

        # Count words
        word_counter = Counter(words)

        self.vocab = word_counter.most_common(len(word_counter))

    def mask_text(self):
        pass
