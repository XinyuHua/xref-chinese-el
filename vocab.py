import os
import utils

class Vocab(object):

    def __init__(self, dict_path):
        assert os.path.exists(dict_path), f"{dict_path} does not exist!"
        self.tokens = []
        self.unk_idx = 1
        self.pad_idx = 0
        self.token2id = dict()

        for ln_id, ln in enumerate(open(dict_path)):
            self.tokens.append(ln.strip())
            self.token2id[ln.strip()] = ln_id

        self.vocab_size = len(self.tokens)
        assert self.token2id[self.tokens[10]] == 10


    def word2id(self, word):
        if word not in self.token2id:
            return self.unk_idx

        return self.token2id[word]


    def id2word(self, idx):
        return self.tokens[idx]

    def __len__(self):
        return self.vocab_size