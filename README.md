# XREF

This repository contains code and links to data for our AKBC 2020 paper:

```
XREF: Entity Linking for Chinese News Comments with Supplementary Article Reference
Xinyu Hua, Lei Li, Lifeng Hua, and Lu Wang. AKBC 2020
```

## Data

coming soon


## Environment

- python 3.7
- pytorch 1.5
- pytorch-lightning 0.8


## Training

The following script will train XRef model for 10 epochs with batch size 32. It will dump checkpoints every 1 epoch, with the top 5 ones (base on validation loss) saved.

```shell script
python train.py --domain=[ent,product] \
    --ckpt-dir=model/demo/ \
    --batch-size=32 \
    --model=xref \
    --tensorboard-dir=demo \
    --save-interval=1 \
    --save-topk=5 \
    --max-train-epochs=10
```

## Inference

The following script loads the latest checkpoint from `--ckpt-dir` and run inference over the test set. The results will be saved to `--output-path`.

```shell script
python infer.py --domain=[ent,product] \
    --output-path=demo.jsonl \
    --ckpt-dir=model/demo/ \
```




## License

See the [LICENSE](LICENSE) file for details.
