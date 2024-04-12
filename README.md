# The Code of EffiBlackVip 
Anonymous ECCV 2024 Submission Paper ID #11795

# Setup
```
cd EffBlackVIP

conda create -y -n effblackvip python=3.8

conda activate effblackvip

conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.6 -c pytorch -c conda-forge

cd my_dassl
pip install -r requirements.txt

cd ..
pip install -r requirements.txt
```

# Data preparation
You can get data from [BlackVIP](https://github.com/changdaeoh/BlackVIP). Then put the data in the `/home/public/BlackVIPData\#dataset` folder. For example, for the SVHN dataset, you should put the data in the `/home/public/BlackVIPData\svhn`

You can get vit-mae-base from [Huggingface](https://huggingface.co/facebook/vit-mae-base). Then put the model in the`/home/public/Pretrains/vit-mae-base` folder

# Run
## Get Pre-train Result
```
cd scripts

cd blackvip

bash run.sh #dataset #gpu_id
#For example, for the SVHN dataset and gpu 1
bash run.sh svhn 1
```
The output is saved in the `pre_output`
## Subspace Learning
```
cd scripts

cd myblackvip

bash run.sh #dataset #gpu_id
#For example, for the SVHN dataset and gpu 1
bash run.sh svhn 1
```
The output is saved in the `output`
## Black-Box Visual Prompt Tuning
```
cd scripts

cd myblackvip

bash run_ft.sh #dataset #gpu_id
#For example, for the SVHN dataset and gpu 1
bash run_ft.sh svhn 1
```
The output is saved in the `output`
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
