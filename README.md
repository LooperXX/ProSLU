# Text is no more Enough! A Benchmark for Profile-based Spoken Language Understanding

This repository contains the official `PyTorch` implementation of the paper:

**Text is no more Enough! A Benchmark for Profile-based Spoken Language Understanding**.

[Xiao Xu*](http://ir.hit.edu.cn/~xxu/), [Libo Qin*](http://ir.hit.edu.cn/~lbqin/), Kaiji Chen, Guoxing Wu, Linlin Li, [Wanxiang Che](http://ir.hit.edu.cn/~car/).

***AAAI 2022***. [[Paper(Arxiv)]](https://arxiv.org/abs/2112.11953) [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/21411)

<div align=center>
  <img src="img/example.jpg" alt="example" width="600pt" height="355pt"/>
</div>

In the following, we will guide you how to use this repository step by step.

## Workflow
<div align=center>
  <img src="img/annotation.jpg" alt="framework">
</div>

## Architecture
<div align=center>
  <img src="img/framework.jpg" alt="framework"/>
</div>

## Results
<div align=center>
  <img src="img/result.jpg" alt="result"/ />
</div>

## Preparation

Our code is based on the following packages:

- numpy==1.19.5
- tqdm==4.50.2
- pytorch==1.7.0
- python==3.7.3
- cudatoolkit==11.0.3
- transformers==4.1.1

We highly suggest you using [Anaconda](https://www.anaconda.com/) to manage your python environment.

We download the chinese pretrained model checkpoints from the following links:
- [BERT & RoBERTa](https://github.com/ymcui/Chinese-BERT-wwm)
- [XLNet](https://github.com/ymcui/Chinese-XLNet)
  - [Huggingface chinese-xlnet-base](https://huggingface.co/hfl/chinese-xlnet-base/)
- [ELECTRA](https://github.com/ymcui/Chinese-ELECTRA) 
  - [Huggingface chinese-electra-180g-base-discriminator](https://huggingface.co/hfl/chinese-electra-180g-base-discriminator/)
  - [Huggingface chinese-electra-180g-base-generator](https://huggingface.co/hfl/chinese-electra-180g-base-generator/)

## How to Run it

The script **train.py** acts as a main function to the project, you can run the experiments by the following commands.

```Shell
# LSTM w/o Profile on TITAN Xp
python train.py -g -fs -es -uf -bs 8 -lr 0.0006
# LSTM w/ Profile on TITAN Xp
python train.py -g -fs -es -uf -ui -bs 8 -lr 0.0004
# BERT w/o Profile on Tesla V100s PCIE 32GB
python train.py -g -fs -es -uf -up -mt XLNet -bs 8 -lr 0.001 -blr 4e-05
# BERT w/ Profile on Tesla V100 PCIE 32GB
python train.py -g -fs -es -uf -up -ui -mt ELECTRA -bs 8 -lr 0.0008 -blr 4e-05
```

If you have any question, please issue the project or email [me](mailto:xxu@ir.hit.edu.cn) or [lbqin](mailto:lbqin@ir.hit.edu.cn), and we will reply you soon.

## Acknowledgement
- We are highly grateful for the public code of Stack-Propagation!
  > A Stack-Propagation Framework with Token-Level Intent Detection for Spoken Language Understanding. Libo Qin,Wanxiang Che, Yangming Li, Haoyang Wen and Ting Liu. *(EMNLP 2019)*. Long paper. [[pdf]](https://www.aclweb.org/anthology/D19-1214/) [[code]](https://github.com/LeePleased/StackPropagation-SLU)
- We are highly grateful for the open-source knowledge graph!
  - [CN-DBpedia](http://kw.fudan.edu.cn/apis/cndbpedia/)
  - [OwnThink](https://www.ownthink.com/docs/kg/)

## Citation
If you use the source code or the dataset in this repository, please cite our paper.
```
@article{Xu_Qin_Chen_Wu_Li_Che_2022,
  title        = {Text Is No More Enough! A Benchmark for Profile-Based Spoken Language Understanding},
  volume       = {36},
  url          = {https://ojs.aaai.org/index.php/AAAI/article/view/21411},
  doi          = {10.1609/aaai.v36i10.21411},
  abstractnote = {Current researches on spoken language understanding (SLU) heavily are limited to a simple setting: the plain text-based SLU that takes the user utterance as input and generates its corresponding semantic frames (e.g., intent and slots). Unfortunately, such a simple setting may fail to work in complex real-world scenarios when an utterance is semantically ambiguous, which cannot be achieved by the text-based SLU models. In this paper, we first introduce a new and important task, Profile-based Spoken Language Understanding (ProSLU), which requires the model that not only relies on the plain text but also the supporting profile information to predict the correct intents and slots. To this end, we further introduce a large-scale human-annotated Chinese dataset with over 5K utterances and their corresponding supporting profile information (Knowledge Graph (KG), User Profile (UP), Context Awareness (CA)). In addition, we evaluate several state-of-the-art baseline models and explore a multi-level knowledge adapter to effectively incorporate profile information. Experimental results reveal that all existing text-based SLU models fail to work when the utterances are semantically ambiguous and our proposed framework can effectively fuse the supporting information for sentence-level intent detection and token-level slot filling. Finally, we summarize key challenges and provide new points for future directions, which hopes to facilitate the research.},
  number       = {10},
  journal      = {Proceedings of the AAAI Conference on Artificial Intelligence},
  author       = {Xu, Xiao and Qin, Libo and Chen, Kaiji and Wu, Guoxing and Li, Linlin and Che, Wanxiang},
  year         = {2022},
  month        = {Jun.},
  pages        = {11575-11585}
}
```