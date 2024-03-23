#cd /aiarena/gpfs/glossification
pip install transformers
pip install tensorboardX
pip install tokenizers
HF_ENDPOINT=https://hf-mirror.com python train.py --share_target_embeddings --use_pre_trained_embedding