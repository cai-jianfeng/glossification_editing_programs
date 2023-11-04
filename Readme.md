# glossification with editing causal attention (GECA)

This is the code implementation of the paper [Transcribing Natural Languages for the Deaf via Neural Editing Programs](https://ojs.aaai.org/index.php/AAAI/article/view/21457) using pytorch based on the [transformer library](https://github.com/tunz/transformer-pytorch).

## List of the code files

- [extract_editing_program.py](/extract_editoring_program.py): This is the implementation of the minimal editing program using **Dynamic Programming** and **Backtracking** algorithm.
- [transformer.py](/transformer.py): This is the implementation of the **Transformer**, including MHA, FFN and so on.
- [model.py](/model.py): This is the implementation of the *GECA*, including Generator, Executor and editing causal attention.
- [optimizer.py](/optimizer.py): This is the encapsulation of the **Adam** optimizer in order to easily use.
- [utils.py](/utils.py): This is the useful utility code, including mask computation, loss computation and so on.
- [train.py](/train.py): This is the main code used to implement **model training**, validation, logging, etc.