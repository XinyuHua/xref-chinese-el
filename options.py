import argparse
import utils

def _add_common_parser(parser):
    parser.add_argument('--ckpt-dir', default=None,
                        help='Path to save/restore checkpoints.')
    parser.add_argument('--domain', type=str, choices=['ent', 'product'],
                        required=True)
    parser.add_argument('--batch-size', type=int, default=32)

    parser.add_argument('--char-vocab-path', type=str,
                        default=utils.DATA_PATH + 'char.dict')
    parser.add_argument('--ent-vocab-path', type=str,
                        default=utils.DATA_PATH + 'entity.dict')

    # model options
    parser.add_argument('--model', type=str, choices=['xref', 'logreg'])
    parser.add_argument('--add-features', action='store_true')
    # parser.add_argument('--activation', type=str, choices=['tanh'])
    parser.add_argument('--hidden-size', type=int, default=100,
                        help='Dimension of the reduced space.')
    parser.add_argument('--rnn-size', type=int, default=100,
                        help='Dimension of the RNN.')

    # parser.add_argument('--')

def _add_train_parser(parser):
    parser.add_argument('--max-train-epochs', type=int, default=10)
    parser.add_argument('--save-interval', type=int, default=1)
    parser.add_argument('--save-topk', type=int, default=5)
    parser.add_argument('--learning-rate', type=float, default=0.0001)
    parser.add_argument('--dropout-prob', type=float, default=0.0)
    parser.add_argument('--max-grad-norm', type=float, default=5.0)
    parser.add_argument('--tensorboard-dir', type=str, default='demo')

def _add_inference_parser(parser):
    parser.add_argument('--output-path', type=str, required=True)
    _add_train_parser(parser)

def get_training_config():
    parser = argparse.ArgumentParser()
    _add_common_parser(parser)
    _add_train_parser(parser)
    return parser

def get_inference_config():
    parser = argparse.ArgumentParser()
    _add_common_parser(parser)
    _add_inference_parser(parser)
    return parser