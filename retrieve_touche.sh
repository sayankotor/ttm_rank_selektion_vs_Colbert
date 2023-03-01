
python3 -m colbert.retrieve --amp --doc_maxlen 180 --mask-punctuation --bsize 32 --queries /notebook/ColBERT/collections/touche21_queries.tsv --nprobe 32 --partitions 50 --faiss_depth 100 --index_root /notebook/ColBERT/indexes --index_name MSMARCO_tt --checkpoint /notebook/ColBERT/regular_checkpoints/folder_with_main_chekpoints/edinburg_colbert.dnn --root /notebook/ColBERT/experiments --experiment MSMARCO-tt

CUDA_VISIBLE_DEVICES="4" python3 -m colbert.index_faiss --index_root /notebook/ColBERT/indexes/ --index_name MSMARCO_tt --root /notebook/ColBERT/experiments/ --experiment MSMARCO-tt --partitions 70

python3 -m colbert.retrieve --amp --doc_maxlen 180 --mask-punctuation --bsize 32 --queries /notebook/ColBERT/collections/touche21_queries.tsv --nprobe 32 --partitions 70 --faiss_depth 1024 --index_root /notebook/ColBERT/indexes --index_name MSMARCO_tt --checkpoint /notebook/ColBERT/regular_checkpoints/folder_with_main_chekpoints/edinburg_colbert.dnn --root /notebook/ColBERT/experiments --experiment MSMARCO-tt