import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch

from dataloader import ToutiaoEntityLinkingDataset
from vocab import Vocab
import numpy as np
from argparse import Namespace


class ELModel(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        if isinstance(hparams, dict):
            hparams = Namespace(**hparams)
        self.hparams = hparams
        self.learning_rate = hparams.learning_rate
        

        self.char_dict = Vocab(dict_path=hparams.char_vocab_path)
        self.entity_dict = Vocab(dict_path=hparams.ent_vocab_path)


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


    def train_dataloader(self):
        train_dataset = ToutiaoEntityLinkingDataset(set_type='train',
                                                    opt=self.hparams,
                                                    char_dict=self.char_dict,
                                                    ent_dict=self.entity_dict,
                                                    is_inference=False,
                                                    is_pretrain=False)

        dataloader = DataLoader(train_dataset, batch_size=self.hparams.batch_size,
                                collate_fn=train_dataset.collater)
        return dataloader

    def val_dataloader(self):
        valid_dataset = ToutiaoEntityLinkingDataset(set_type='valid',
                                                    opt=self.hparams,
                                                    char_dict=self.char_dict,
                                                    ent_dict=self.entity_dict,
                                                    is_inference=False,
                                                    is_pretrain=False)

        dataloader = DataLoader(valid_dataset, batch_size=self.hparams.batch_size,
                                collate_fn=valid_dataset.collater)
        return dataloader

    def test_dataloader(self):
        test_dataset = ToutiaoEntityLinkingDataset(set_type='test',
                                                    opt=self.hparams,
                                                    char_dict=self.char_dict,
                                                    ent_dict=self.entity_dict,
                                                    is_inference=False,
                                                    is_pretrain=False)

        dataloader = DataLoader(test_dataset, batch_size=self.hparams.batch_size,
                                collate_fn=test_dataset.collater)
        return dataloader

    def training_step(self, batch, batch_idx):
        (loss, output, acc) = self(batch)
        tboard_logs = {'loss': loss,
                       'batch_accuracy': np.mean(acc)}
        return {'loss': loss, 'log': tboard_logs}

    def validation_step(self, batch, batch_idx):
        (loss, output, acc) = self(batch)
        tboard_logs = {'loss': loss,
                       'batch_accuracy': np.mean(acc)}
        return {'val_loss': loss, 'log': tboard_logs, 'val_acc': acc}

    def test_step(self, batch, batch_idx):
        (loss, output, acc) = self(batch)
        tboard_logs = {'test_loss': loss}
        return {'loss': loss, 'log': tboard_logs}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_val_acc = []
        for ln in outputs:
            avg_val_acc.extend(ln['val_acc'])
        avg_val_acc = np.mean(avg_val_acc)

        tensorboard_logs = {'val_loss': avg_loss, 'avg_val_acc': avg_val_acc}
        return {'val_loss': avg_loss, 'progress_bar': tensorboard_logs,
                'log': tensorboard_logs}