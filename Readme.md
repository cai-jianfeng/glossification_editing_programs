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
pip install rouge
```

# preparation

download CSL dataset in ```./``` (under the current project directory) from [huggingface](https://huggingface.co/datasets/caijanfeng/CSL_edit_dataset). 
The dataset file directory structure is as follows:

```
CSL_data
|-- CSL-Daily.txt
|-- CSL-Daily_editing_chinese.txt
|-- CSL-Daily_editing_chinese_past.txt
|-- CSL-Daily_editing_chinese_test.txt
```

# train

train with the editing casual mask and the Executor:

```shell
python train.py --share_target_embeddings --use_pre_trained_embedding
```

train without the editing casual mask and the Executor:

```shell
python train_wo_edit_casual_mask.py --share_target_embeddings --use_pre_trained_embedding
```

# inference

download model checkpoints trained on CSL dataset in ```./``` (under the current project directory) from [huggingface](https://huggingface.co/caijanfeng/Edit-Att)
The checkpoints file directory structure is as follows:

```
|--output_wo_mask
    |-- models
        |-- best_model.pt
        |-- global_step.pt
        |-- last_model.pt
|--output
    |-- models
        |-- best_model.pt
        |-- global_step.pt
        |-- last_model.pt
```

then you can use the model checkpoints to inference any input sentence:

```shell
python inference.py --input=<the input sentence you want to inference> --max_output_len=<the max output length of predicted editing program>
```

```shell
python inference_wo_edit_casual_mask.py --input=<the input sentence you want to inference> --max_output_len=<the max output length of predicted editing program>
```

|                 Methods                  |    BLEU3    |    BLEU4    |   ROUGE-L   |
|:----------------------------------------:|:-----------:|:-----------:|:-----------:|
|             Edit-Att(Origin)             |    24.93    |    18.07    |    49.66    |
|   Edit-Att+$\mathcal{L}_{RL}$(Origin)    |    25.51    |    18.89    |    49.91    |
| Edit-Att_wo_edit_casual_mask(Reproduced) |    21.61    |    17.29    |    63.44    |
|           Edit-Att(Reproduced)           |  **25.66**  |  **22.47**  |  **73.49**  |
| Edit-Att+$\mathcal{L}_{RL}$(Reproduced)  |     ——      |     ——      |     ——      |