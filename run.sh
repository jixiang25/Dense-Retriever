#!/bin/bash

python -m preprocess.preprocess_msmarco_passage_data \
    --official_data_dir=./data/msmarco-passage/official_data \
    --tripplets_and_qrels_dir=./data/msmarco-passage/cls_L2_CE_lr1e-5/tripplets_and_qrels \
    --tokenize_dir=./data/msmarco-passage/cls_L2_CE_lr1e-5/tokenize \
    --collection_memmap_dir=./data/msmarco-passage/cls_L2_CE_lr1e-5/collection_memmap \
    --preprocess_eval_collection \
    --preprocess_complete_collection \
    --generate_tripplets \
    --generate_qrels

python -m train.train_dual_encoder \
    --collection_memmap_dir=./data/msmarco-passage/cls_L2_CE_lr1e-5/collection_memmap/complete \
    --tokenize_dir=./data/msmarco-passage/cls_L2_CE_lr1e-5/tokenize \
    --tripplets_and_qrels_dir=./data/msmarco-passage/cls_L2_CE_lr1e-5/tripplets_and_qrels \
    --checkpoint_dir=./data/msmarco-passage/cls_L2_CE_lr1e-5/checkpoints \
    --logging_dir=./data/msmarco-passage/cls_L2_CE_lr1e-5/log \
    --device=cuda:3 \
    --train_epochs=1 \
    --gradient_accumulate_steps=6 \
    --batch_size_per_gpu=20 \
    --learning_rate=1e-5 \
    --repr_type=cls \
    --similarity_type=L2 \
    --loss=cross-entropy

python -m eval.eval_dual_encoder \
    --do_precompute \
    --load_model_path=./data/msmarco-passage/cls_L2_CE_lr1e-5/checkpoints/step-40000 \
    --collection_memmap_dir=./data/msmarco-passage/cls_L2_CE_lr1e-5/collection_memmap/complete \
    --embedding_dir=./data/msmarco-passage/cls_L2_CE_lr1e-5/embeddings \
    --offline_precompute_device=cuda:0 \
    --offline_precompute_batch_size_per_gpu=40


python -m eval.eval_dual_encoder \
    --do_retrieve \
    --load_model_path=./data/msmarco-passage/cls_L2_CE_lr1e-5/checkpoints/step-40000 \
    --tokenize_dir=./data/msmarco-passage/cls_L2_CE_lr1e-5/tokenize \
    --embedding_dir=./data/msmarco-passage/cls_L2_CE_lr1e-5/embeddings \
    --retrieved_result_dir=./data/msmarco-passage/cls_L2_CE_lr1e-5/retrieved_result \
    --online_retrieve_device=cuda:0 \
    --online_retrieve_batch_size=100 \
    --hit_num=1000 \
    --docs_per_gpu=1500000

python -m metrics.calculate_mrr10 \
    --path_to_reference=./data/msmarco-passage/official_data/qrels.dev.small.tsv \
    --path_to_candidate=./data/msmarco-passage/cls_L2_CE_lr1e-5/retrieved_result/top1000.rank.txt