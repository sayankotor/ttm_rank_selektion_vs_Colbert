#Retrieve and evaluation on MSMARCO data


python -m colbert.test --amp --doc_maxlen 180 --mask-punctuation \
--collection /notebook/ColBERT/collections/MSMARCO/collection.tsv \
--queries /notebook/ColBERT/collections/MSMARCO/queries.dev.small.tsv \
--topk /notebook/ColBERT/collections/MSMARCO/top1000.dev  \
--checkpoint /notebook/ColBERT/regular_checkpoints/folder_with_main_chekpoints/edinburg_colbert.dnn \
--root /root/to/experiments/ --experiment MSMARCO_eval  --qrels /notebook/ColBERT/collections/MSMARCO/qrels.dev.small.tsv --is_compressed --compressed_type ttm_


CUDA_VISIBLE_DEVICES="4,5" python3 -m colbert.index --amp --doc_maxlen 180 --mask-punctuation --bsize 256 --checkpoint /notebook/ColBERT/regular_checkpoints/folder_with_main_chekpoints/edinburg_colbert.dnn --collection /notebook/ColBERT/collections/MSMARCO/collection.tsv --index_root /notebook/ColBERT/indexes/ --index_name MSMARCO_eval_full --root /notebook/ColBERT/experiments/ --experiment MSMARCO_eval_full --is_compressed --compressed_type ttm_custom