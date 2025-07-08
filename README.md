# HemoGAT: Heterogeneous multi-modal emotion recognition with cross-modal transformer and graph attention network
<i>
  Official code repository for the manuscript 
  <b>"HemoGAT: Heterogeneous Multi-Modal Emotion Recognition with Cross-Modal Transformer and Graph Attention Network"</b>, 
  submitted to 
  <a href="https://advances.vsb.cz/">Advances in Electrical and Electronic Engineering</a>.
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
[**How to run**](#how-to-run) •
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
## How to run

<p>
  This section provides step-by-step instructions to <strong>extract features, train, and predict</strong> using <strong>HemoGAT</strong>.
</p>

<h3>1. Download HemoGAT Resources</h3>
<p>
  The pre-extracted data samples, pretrained models, and configuration files are available at:
</p>
<p>
  <a href="https://drive.google.com/drive/folders/187ARizXiEco3Cwz97eroW8QHkZyZxh2g?usp=drive_link" target="_blank" rel="noopener noreferrer">
    Download HemoGAT Resources
  </a>
</p>
<p>
  Download and extract the resources to your workspace before proceeding.
</p>

<h3>2. Feature Extraction</h3>
<p>
  To extract text and audio embeddings using BERT and wav2vec2, run:
</p>
<pre><code>python feature_extract/BERT_wav2vec2.py
</code></pre>
<p>
  By default, this will load your dataset, extract BERT-based text embeddings and wav2vec2-based audio embeddings, and save them into a <code>feature</code> directory.
</p>

<h3>3. Training the Model</h3>
<p>
  To train HemoGAT on your extracted features, use:
</p>
<pre><code>python main.py --data_dir Path/to/feature/folder --dataset MELD --num_classes 7 --k_text 2 --k_audio 8
</code></pre>
<p>
  <strong>Arguments:</strong>
  <ul>
    <li><code>--data_dir</code>: Path to the extracted feature folder.</li>
    <li><code>--dataset</code>: Dataset to train on (e.g., MELD, IEMOCAP).</li>
    <li><code>--num_classes</code>: Number of emotion classes.</li>
    <li><code>--k_text</code>: Number of neighbors for the text graph.</li>
    <li><code>--k_audio</code>: Number of neighbors for the audio graph.</li>
  </ul>
</p>

<h3>4. Prediction</h3>
<p>
  To predict using the trained HemoGAT model, run:
</p>
<pre><code>python predict.py --data_dir feature --dataset MELD --num_classes 7 --k_text 2 --k_audio 8
</code></pre>
<p>
  This will load your model checkpoint and output predicted emotion labels along with evaluation metrics.
</p>

## References
[1] Nhat Truong Pham, SERVER: Multi-modal Speech Emotion Recognition using Transformer-based and Vision-based Embeddings (ICIIT), 2023. Available https://github.com/nhattruongpham/mmser.git.

[2] Mustaqeem Khan, MemoCMT: Cross-Modal Transformer-Based Multimodal Emotion Recognition System (Scientific Reports), 2025. Available https://github.com/tpnam0901/MemoCMT.

[3] Nhut Minh Nguyen, FleSER: Multi-modal emotion recognition via dynamic fuzzy membership and attention fusion, 2025. Available https://github.com/aita-lab/FleSER.

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
