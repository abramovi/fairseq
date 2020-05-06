#!/bin/bash

# rm -f -r /cnvrg/experimental
# mkdir /cnvrg/experimental
# mv /cnvrg/examples/language_model/wikitext-103 /cnvrg/experimental/wikitext-103
# rm -f -r /cnvrg/experimental/checkpoints
# mkdir /cnvrg/experimental/checkpoints

TEXT=/cnvrg/experimental/wikitext-103
fairseq-preprocess \
    --only-source \
    --trainpref $TEXT/wiki.train.tokens \
    --validpref $TEXT/wiki.valid.tokens \
    --testpref $TEXT/wiki.test.tokens \
    --destdir /cnvrg/experimental/data-bin/wikitext-103 \
    --workers 20

exit

fairseq-train --task language_modeling \
    /cnvrg/experimental/wikitext-103 \
    --save-dir /cnvrg/experimental/transformer_wikitext-103 \
    --arch fconv_lm_dauphin_wikitext103 \
    --max-epoch 35 \ --optimizer nag \
    --lr 1.0 --lr-scheduler reduce_lr_on_plateau --lr-shrink 0.5 \
    --clip-norm 0.1 --dropout 0.2 --weight-decay 5e-06 --criterion adaptive_loss \
    --adaptive-softmax-cutoff 10000,20000,200000 --max-tokens 1024 --tokens-per-sample 1024 \
    --ddp-backend=no_c10d

# Train transormer
# fairseq-train --task language_modeling \
#   /cnvrg/experimental/wikitext-103 \
#   --save-dir /cnvrg/experimental/transformer_wikitext-103 \
#   --arch transformer_lm --share-decoder-input-output-embed \
#   --dropout 0.1 \
#   --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 0.0 \
#   --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
#   --tokens-per-sample 512 --sample-break-mode none \
#   --max-tokens 2048 --update-freq 16 \
#   --fp16 \
#   --max-update 50000

  