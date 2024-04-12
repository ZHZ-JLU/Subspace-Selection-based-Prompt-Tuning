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
```
## Subspace Selection
```
```
## Prompt Tuning
```

```
# Citation
```
@inproceedings{sun2022bbt,
  title={Black-Box Tuning for Language-Model-as-a-Service}, 
  author={Tianxiang Sun and Yunfan Shao and Hong Qian and Xuanjing Huang and Xipeng Qiu},
  booktitle = {Proceedings of {ICML}},
  year={2022}
}
@inproceedings{sun2022bbtv2,
  title={BBTv2: Towards a Gradient-Free Future with Large Language Models},
  author={Tianxiang Sun and Zhengfu He and Hong Qian and Yunhua Zhou and Xuanjing Huang and Xipeng Qiu},
  booktitle = {Proceedings of {EMNLP}},
  year={2022}
}
```
# Acknowledgements 
Our overall experimental pipeline is based on [BBT](https://github.com/txsun1997/Black-Box-Tuning) repository.
