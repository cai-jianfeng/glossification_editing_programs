#cd /aiarena/gpfs/glossification
pip install transformers
pip install tensorboardX
pip install tokenizers
pip install nltk
pip install rouge
#HF_ENDPOINT=https://hf-mirror.com python train_wo_edit_casual_mask.py --share_target_embeddings --use_pre_trained_embedding

HF_ENDPOINT=https://hf-mirror.com python train.py --share_target_embeddings --use_pre_trained_embedding