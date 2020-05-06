#!/bin/bash

# rm -f -r /cnvrg/experimental
# mkdir /cnvrg/experimental
# mv /cnvrg/examples/language_model/wikitext-103 /cnvrg/experimental/wikitext-103
# rm -f -r /cnvrg/experimental/checkpoints
# mkdir /cnvrg/experimental/checkpoints

# TEXT=/cnvrg/experimental/wikitext-103
# fairseq-preprocess \
#     --only-source \
#     --trainpref $TEXT/wiki.train.tokens \
#     --validpref $TEXT/wiki.valid.tokens \
#     --testpref $TEXT/wiki.test.tokens \
#     --destdir /cnvrg/experimental/data-bin/wikitext-103 \
#     --workers 20

# Train
fairseq-train --task language_modeling \
  /cnvrg/experimental/data-bin/wikitext-103 \
  --save-dir /cnvrg/experimental/checkpoints \
  --arch transformer_lm --share-decoder-input-output-embed \
  --dropout 0.1 \
  --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 0.0 \
  --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
  --tokens-per-sample 512 --sample-break-mode none \
  --max-tokens 2048 --update-freq 16 \
  --fp16 \
  --max-update 50000

