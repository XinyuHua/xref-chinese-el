import os
import time
import logging
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from options import get_training_config
from models import XRef
import utils

logger = logging.getLogger(__name__)

class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        logger.info("***** Validation results *****")
        metrics = trainer.callback_metrics
        # Log results
        for key in sorted(metrics):
            if key not in ["log", "progress_bar"]:
                logger.info("{} = {}\n".format(key, str(metrics[key])))

    def on_test_end(self, trainer, pl_module):
        logger.info("***** Test results *****")
        metrics = trainer.callback_metrics

        # Log and save results to file
        output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
        with open(output_test_results_file, "w") as writer:
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    logger.info("{} = {}\n".format(key, str(metrics[key])))
                    writer.write("{} = {}\n".format(key, str(metrics[key])))


def main():
    parser = get_training_config()
    args = parser.parse_args()
    if args.model == 'xref':
        model_class = XRef
    else:
        raise NotImplementedError(f'{args.model} not implemented.')

    if os.path.exists(args.ckpt_dir) and os.listdir(args.ckpt_dir):
        latest_ckpt_path = utils.get_latest_ckpt_path(args.ckpt_dir)
        logger.info(f'Loading checkpoint {latest_ckpt_path}')
        new_epoch_id = os.path.basename(latest_ckpt_path).split('_')[0][-2:]
        new_epoch_id = int(new_epoch_id)
        model = model_class.load_from_checkpoint(latest_ckpt_path)
        model.current_epoch = new_epoch_id + 1
        logger.info(f'Resuming training from epoch {model.current_epoch}')

    else:
        model = model_class(args)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=os.path.join(args.ckpt_dir, "{epoch:02d}_{val_loss:.4f}"),
        monitor="val_loss", mode="min",
        save_top_k=args.save_topk,
        period=args.save_interval,
        save_weights_only=False,
    )

    train_params = dict(
        gpus=1,
        max_epochs=args.max_train_epochs,
        early_stop_callback=False,
        gradient_clip_val=args.max_grad_norm,
        checkpoint_callback=checkpoint_callback,
        callbacks=[LoggingCallback()],
    )

    train_params['logger'] = TensorBoardLogger('tboard',
                                               name=f"{args.tensorboard_dir}_{time.strftime('%Y%m%d_%H%M%S')}")
    trainer = pl.Trainer(**train_params)
    trainer.current_epoch = model.current_epoch
    trainer.fit(model)


if __name__ == '__main__':
    main()