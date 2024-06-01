# Results of supplementary experiments and the Code of Paper
Anonymous KDD 2024 Submission Paper

# Setup
```
conda create --name slt python=3.8
conda activate slt
pip install transformers==4.1.1
pip install fastNLP==0.6.0
pip install datasets
pip install cma
pip install sklearn
```
# Run
## Subspace Learning
```
python subspace_learn.py --task_name sst2 --seed 42 --device 'cuda:0' --learn_dim 10 > ./sst2_learn_42.out
```
## Subspace Selection
```
python subspace_selection.py --task_name sst2 --intrinsic_dim 10 --device 'cuda:0' --seed 42 --train_batch_size 32 --topk 2 > ./sst2_selection_42.out

```
## Prompt Tuning
```
python subspace_op.py --task_name sst2 --seed 42 --device 'cuda:0' --train_batch_size 32 --intrinsic_dim 2 --idx_list 0 1 > ./sst2_op_42.out
```
# Acknowledgements 
Our overall experimental pipeline is based on [BBT](https://github.com/txsun1997/Black-Box-Tuning) repository.
