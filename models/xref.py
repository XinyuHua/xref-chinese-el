import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from models import ELModel
import utils

class XRef(ELModel):

    def __init__(self, hparams):
        super().__init__(hparams)

        # for candidate entity, calculate node2vec embedding and
        # entity-word embeddings
        self.node2vec_embed = nn.Embedding(len(self.entity_dict), 300,
                                           padding_idx=0)
        n2v_pret = np.load(utils.DATA_PATH + 'entity.node2vec.npz')
        n2v_pret[0] = 0.
        self.node2vec_embed.weight.data.copy_(torch.from_numpy(n2v_pret))

        self.ent_word_embed = nn.Embedding(len(self.entity_dict), 300,
                                           padding_idx=0)
        svd_pret = np.load(utils.DATA_PATH + 'entity.svd.npz')
        svd_pret[0] = 0.
        self.ent_word_embed.weight.data.copy_(torch.from_numpy(svd_pret))


        # transform the concatenation of entity embeddings into
        # a d-dimensional space
        self.ent_transform = nn.Linear(2 * 300, self.hparams.hidden_size)
        self.activation = nn.Tanh()


        # character embedding is used for comment LSTM
        self.char_embed = nn.Embedding(len(self.char_dict), 300, padding_idx=0)
        char_pret = np.load(utils.DATA_PATH + 'char.emb.npz')
        char_pret[0] = 0.
        self.char_embed.weight.data.copy_(torch.from_numpy(char_pret))


        self.comment_lstm = nn.LSTM(input_size=1 + 300,
                                    hidden_size=self.hparams.rnn_size,
                                    num_layers=1,
                                    bias=True,
                                    batch_first=True,
                                    dropout=self.hparams.dropout_prob,
                                    bidirectional=True)
        # h_i^T x W_c x \tilde{m}, where h_i^T \in R^{rnn_size x 2}, \tilde{m} \in R^{300}
        self.comment_attn_matrix = nn.Linear(300, self.hparams.rnn_size * 2)

        # transform the concatenation of article attention, comment
        # attention, and comment LSTM repr into mention representatio
        self.comment_transform = nn.Linear(300 * 2 + self.hparams.rnn_size * 2 + self.hparams.rnn_size * 2,
                                           self.hparams.hidden_size)

        self.article_attn_matrix = nn.Linear(300, 300 * 2)

        self.output_layer = nn.Linear(22, 1)

        self.criterion = nn.BCEWithLogitsLoss()


    def _compute_entity_repr(self, cand_ent_ids):
        """Calculate entity representations for all candidates.

        u^{nod} (bsz x 300): node2vec embedding
        u^{wrd} (bsz x 300): svd embedding
        u = [u^{nod}; u^{wrd}] (bsz x 600)
        entity_repr = tanh(W_e * u + b_e) (bsz x 100)
        """

        u_nod = self.node2vec_embed(cand_ent_ids)
        u_wrd = self.ent_word_embed(cand_ent_ids)
        u_cat = torch.cat((u_nod, u_wrd), dim=1)

        entity_repr = self.activation(self.ent_transform(u_cat))

        return entity_repr

    def _compute_mention_repr(self, comment_char_ids, mention_ind, article_ent_ids):
        """Calculate mention representation in three parts:
        1) mention base v^{base} (bsz x 200): the last hidden state of comment RNN
        2) comment attention v^{cmt}
        3) article attention v^{art}

        v_m = tanh(W * [v^{base}; v^{cmt}; v^{art}] + b) (bsz x 100)
        """
        comment_embed = self.char_embed(comment_char_ids)
        comment_len = comment_char_ids.ne(0).sum(1)

        ment_ind = mention_ind.float()
        comment_rnn_input = torch.cat((comment_embed, ment_ind.unsqueeze(-1)), dim=2)
        comment_rnn_input_packed = pack(comment_rnn_input, comment_len,
                                        batch_first=True, enforce_sorted=False)

        comment_rnn_mem_bank, (comment_rnn_final, _) \
            = self.comment_lstm(comment_rnn_input_packed)
        comment_rnn_final = torch.cat((comment_rnn_final[0], comment_rnn_final[1]), dim=1)
        comment_rnn_mem_bank = unpack(comment_rnn_mem_bank, batch_first=True)[0]

        # calculate comment attention
        # use average mention character embeddings to attend the memory bank
        # of comment LSTM
        interm = ment_ind.float().unsqueeze(-1) * comment_embed
        interm_avg = interm.sum(-2) / ment_ind.sum(-1).reshape(-1, 1)
        # DEBUG: interm_avg should be (bsz x char_emb_dim)

        query = self.comment_attn_matrix(interm_avg)
        # DEBUG: query (bsz x 200)

        key = comment_rnn_mem_bank
        # DEBUG: key (bsz x max_len x 200)

        comment_attn_scores = (key * query.unsqueeze(1)).sum(-1)
        # DEBUG: scores (bsz x max_len x 1)
        comment_attn = nn.Softmax(dim=-1)(comment_attn_scores)
        # DEBUG: attn (bsz x max_len)
        comment_attn_repr = (comment_attn.unsqueeze(-1) * comment_rnn_mem_bank).sum(1)
        # DEBUG: attn_repr (bsz x rnn_size)

        # art_ent_repr (bsz x 600)
        art_ent_emb = torch.cat((self.node2vec_embed(article_ent_ids),
                                 self.ent_word_embed(article_ent_ids)), 2)
        art_attn_query = self.article_attn_matrix(interm_avg)
        art_attn_key = art_ent_emb
        art_attn_scores = (art_attn_key * art_attn_query.unsqueeze(1)).sum(-1)
        art_attn = nn.Softmax(dim=-1)(art_attn_scores)
        art_ent_repr = (art_attn.unsqueeze(-1) * art_ent_emb).sum(1)

        # Mention representation
        # v_m = [v_m^{base}; v_m^{cmt}; v_m^{art}]
        # v_m (bsz x (200 + 200 + 600))
        mention_repr = torch.cat((comment_rnn_final, comment_attn_repr, art_ent_repr), dim=1)
        mention_repr = self.activation(self.comment_transform(mention_repr))
        return mention_repr


    def forward(self, batch):

        mention_repr = self._compute_mention_repr(batch['comment'],
                                                  batch['mention_ind'],
                                                  batch['art_ref_ent'])
        entity_repr = self._compute_entity_repr(batch['cand_ent'])

        # Calculate dot-product between two reprs
        inner_prod = (entity_repr * mention_repr).sum(1)
        # append features
        logits = self.output_layer(torch.cat((inner_prod.unsqueeze(1), batch['features']), 1)).squeeze()

        output_probs = nn.Sigmoid()(logits)
        output = (output_probs,)

        if 'labels' in batch:
            loss = self.criterion(logits, batch['labels'])
            instance_accuracy = ((output_probs > 0.5) == batch['labels']).long()
            output = (loss, output, instance_accuracy.tolist())
        return output