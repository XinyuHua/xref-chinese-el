import numpy as np
import time
import json
import utils

import torch
from torch.utils.data import Dataset

np.random.seed(0)

DATA_PATH = "/data/model/xinyu/akbc2020_release/to_release/"

class ToutiaoEntityLinkingDataset(Dataset):

    def __init__(self, set_type, opt, char_dict, ent_dict,
                 is_inference=False, is_pretrain=False):
        super().__init__()
        self.set_type = set_type
        self.domain = opt.domain
        self.is_inference = is_inference
        self.is_pretrain = is_pretrain
        self.instances = []
        self.num_comments = []
        self.num_mentions = 0

        self.char_dict = char_dict
        self.ent_dict = ent_dict

        self.load_raw_dataset()

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        instance = self.instances[index]
        ret_obj = dict(
            cmt_text=instance[0],
            cand_text=instance[3],
            cmt_ids=instance[1],
            ment_ind=instance[2],
            cand_ent=instance[4],
            features=instance[5],
            art_ref_ent=instance[6],
            label=instance[7],
            id=instance[8],
            mention_tuple=instance[9],
        )
        return ret_obj

    def collater(self, samples):
        """Convert a list of instances into a batch, do padding for comment"""
        def merge(key, is_list=False, pad_idx=0):
            if is_list:
                res = []
                for i in range(len(samples[0][key])):
                    res.append(utils.collate_tokens(
                        [s[key][i] for s in samples], pad_idx=pad_idx,
                    ))
                return res
            else:
                return utils.collate_tokens([s[key] for s in samples],
                                            pad_idx=pad_idx)

        ret_obj = dict(
            id=[s['id'] for s in samples],
            comment=merge('cmt_ids', pad_idx=0),
            mention_ind=merge('ment_ind', pad_idx=0),
            features=merge('features'),
            cand_ent=torch.LongTensor([s['cand_ent'] for s in samples]),
            art_ref_ent=merge('art_ref_ent'),
            labels=torch.FloatTensor([int(s['label']) for s in samples]),
            comment_text=[s['cmt_text'] for s in samples],
            cand_text=[s['cand_text'] for s in samples],
            mention_tuple=[s['mention_tuple'] for s in samples],
        )
        return ret_obj


    def load_raw_dataset(self):

        t0 = time.time()
        pos_labels = 0

        jsonl_path = DATA_PATH + f'{self.domain}_'
        if self.is_pretrain:
            jsonl_path += 'unlabeled.jsonl'
        else:
            jsonl_path += 'labeled.jsonl'

        art_ref_entity_path = DATA_PATH + f'{self.domain}_art_ref_entity.jsonl'
        art_ref_entity = [json.loads(ln) for ln in open(art_ref_entity_path)]

        features_path = DATA_PATH + f'features/{self.domain}_labeled.jsonl'
        features = [json.loads(ln) for ln in open(features_path)]

        for ln_id, ln in enumerate(open(jsonl_path)):
            cur_feat = features[ln_id]['comments']
            cur_obj = json.loads(ln)
            if cur_obj['split'] != self.set_type: continue

            self.num_comments.append(len(cur_obj['comments']))
            cur_art_ref_entity_ids = [self.ent_dict.word2id(e) for e in art_ref_entity[ln_id]]


            for cid, cmt in enumerate(cur_obj['comments']):
                cmt_text = cmt['text']
                cmt_char_ids = [self.char_dict.word2id(w) for w in cmt['text']]

                assert cur_feat[cid]['comment'] == cmt_text
                self.num_mentions += len(cmt['mentions'])

                for ment_ix, ment in enumerate(cmt['mentions']):
                    ment_ind = np.zeros([len(cmt_text)], dtype=np.int)
                    ment_ind[ment['span'][0]: ment['span'][1]] = 1
                    ment_feat = cur_feat[cid]['features'][ment_ix]

                    for cand_ix, cand in enumerate(ment['candidates']):
                        if not cand in ment_feat: continue

                        cand_id = self.ent_dict.word2id(cand)
                        instance_feats = ment_feat[cand]
                        # comment_text, comment_char_ids, mention indicator, candidate_text, candidate_id, features, article_ent, label, id
                        label = cand in ment['labels']
                        if label:
                            pos_labels += 1
                        instance_id = f'{cur_obj["id"]}_{cmt["cid"]}_{ment_ix}_{cand_ix}'

                        cmt_char_ids = torch.LongTensor(cmt_char_ids)
                        ment_ind = torch.LongTensor(ment_ind)
                        instance_feats = torch.Tensor(instance_feats)
                        art_ref_ent = torch.LongTensor(cur_art_ref_entity_ids)


                        cur_instance = (cmt_text, cmt_char_ids, ment_ind, cand,
                                        cand_id, instance_feats, art_ref_ent, label,
                                        instance_id, (ment['text'], ment['span']))
                        self.instances.append(cur_instance)


        print('{} instances loaded in {:.2f} seconds'.format(len(self.instances), time.time() - t0))
        print('pos vs. neg = 1 vs. {:.2f}'.format((len(self.instances) - pos_labels)/pos_labels))
        np.random.shuffle(self.instances)
