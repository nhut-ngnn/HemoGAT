# HemoGAT: Heterogeneous multi-modal emotion recognition with cross-modal transformer and graph attention network
<i>
  Official code repository for the manuscript 
  <b>"HemoGAT: Heterogeneous Multi-Modal Emotion Recognition with Cross-Modal Transformer and Graph Attention Network"</b>, 
  submitted to 
  <a href="https://advances.vsb.cz/">Journal Articles in Electrical and Electronic Engineering</a>.
</i>

> Please press ⭐ button and/or cite papers if you feel helpful.

<p align="center">
<img src="https://img.shields.io/github/stars/nhut-ngnn/HemoGAT">
<img src="https://img.shields.io/github/forks/nhut-ngnn/HemoGAT">
<img src="https://img.shields.io/github/watchers/nhut-ngnn/HemoGAT">
</p>

<div align="center">

[![python](https://img.shields.io/badge/-Python_3.8.20-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![pytorch](https://img.shields.io/badge/Torch_2.0.1-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![cuda](https://img.shields.io/badge/-CUDA_11.8-green?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit-archive)
</div>

<p align="center">
<img src="https://img.shields.io/badge/Last%20updated%20on-19.04.2025-brightgreen?style=for-the-badge">
<img src="https://img.shields.io/badge/Written%20by-Nguyen%20Minh%20Nhut-pink?style=for-the-badge"> 
</p>


<div align="center">

[**Abstract**](#Abstract) •
[**Install**](#install) •
[**Usage**](#usage) •
[**References**](#references) •
[**Citation**](#citation) •
[**Contact**](#Contact)

</div>

## Abstract 
> Multi-modal speech emotion recognition (SER) is promising, but fusing diverse information streams remains challenging. Sophisticated architectures are required to synergistically combine the modeling of structural relationships across modalities with fine-grained, feature-level interactions. To address this, we introduce HemoGAT, a novel heterogeneous multi-modal SER architecture integrating a cross-modal transformer (CMT) and a graph attention network. HemoGAT employs a dual-stream architecture with two core modules: a heterogeneous multi-modal graph attention network (HM-GAT), which models complex structural and contextual dependencies using a graph of deep embeddings, and a CMT, which enables fine-grained feature fusion through bidirectional cross-attention. This design captures both high-level relationships and immediate inter-modal influences. HemoGAT achieves a 0.29\% improvement in accuracy compared to the previous best on the IEMOCAP dataset, and obtains highly competitive results on the MELD dataset, demonstrating its effectiveness compared to the existing methods. Comprehensive ablation studies evaluate the impact of the Top-K algorithm for heterogeneous graph construction, compare uni-modal and multi-modal fusion strategies, assess the contributions of the HM-GAT and the CMT modules, and analyze the effect of GAT layer depth. 
>
> Index Terms: Heterogeneous graph construction, Graph attention network, Cross-modal transformer, Feature fusion, Multi-modal speech emotion recognition.


## Install
### Clone this repository
```
git clone https://github.com/nhut-ngnn/HemoGAT.git
```

### Create Conda Enviroment and Install Requirement
Navigate to the project directory and create a Conda environment:
```
cd HemoGAT
conda create --name hemogat python=3.8
conda activate hemogat
```
### Install Dependencies
```
pip install -r requirements.txt
```


## References

## Citation
If you use this code or part of it, please cite the following papers:
```
Update soon
```
## Contact
For any information, please contact the main author:

Nhut Minh Nguyen at FPT University, Vietnam

**Email:** [minhnhut.ngnn@gmail.com](mailto:minhnhut.ngnn@gmail.com)<br>
**ORCID:** <link>https://orcid.org/0009-0003-1281-5346</link> <br>
**GitHub:** <link>https://github.com/nhut-ngnn/</link>