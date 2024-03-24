# glossification with editing causal attention (Edit-Att)

This is the code implementation of the paper [Transcribing Natural Languages for the Deaf via Neural Editing Programs](https://ojs.aaai.org/index.php/AAAI/article/view/21457) using pytorch based on the [transformer library](https://github.com/tunz/transformer-pytorch).

# installation

create conda environment and install dependence package

```shell
conda create -n editatt python=3.8
conda activate editatt
pip3 install torch torchvision torchaudio

pip install transformers
pip install tensorboardX
pip install tokenizers
pip install nltk
```

# preparation

download CSL dataset in ```./``` (under the current project directory) from [huggingface](https://huggingface.co/). 
The dataset file directory structure is as follows:

```
CSL_data
|-- CSL-Daily.txt
|-- CSL-Daily_editing.txt
|-- CSL-Daily_editing_chinese.txt
|-- CSL-Daily_editing_chinese_test.txt
|-- CSL-Daily_test.txt
|-- editing_casual_mask_CSL_174_40.npy
|-- editing_casual_mask_CSL_174_40_test.npy
```

# train

```shell
python train.py --share_target_embeddings --use_pre_trained_embedding
```