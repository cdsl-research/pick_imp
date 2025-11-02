# pick_imp

## 構成
```
~/hyouka/information/
├── data
├── anser/
│   ├── answer_<日時>.txt
├── pick_imp.py
```

## 実行する環境
- Python verion：3.12.3

## プログラムの概要
- このシステムは，文章中から 命令・依頼 に当たる文を抽出するプログラムです．
- 多言語対応の自然言語推論モデル mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 を用いて文をゼロショット分類し，
各文に対して「命令・依頼」「説明」「質問」「状況」の確率スコアを算出します．
- スコアが設定した閾値を超えた文を抽出し，結果を anser/ ディレクトリに保存します．

### 環境の構築
- Pythonの仮想環境とtransformers，torchライブラリを導入します．

仮想環境の作成
```
arita@sre000h:~/pick_imp$ python3 -m venv venv
arita@sre000h:~/pick_imp$ 
```

仮想環境の有効化
```
arita@sre000h:~/pick_imp$ source venv/bin/activate
(venv) arita@sre000h:~/pick_imp$ 
```
torch，transformersモジュールのインストール
```
(venv) arita@sre000h:~/pick_imp$ pip install torch transformers
Collecting torch
  Using cached torch-2.9.0-cp312-cp312-manylinux_2_28_x86_64.whl.metadata (30 kB)
Collecting transformers
  Using cached transformers-4.57.1-py3-none-any.whl.metadata (43 kB)
Collecting filelock (from torch)
  Using cached filelock-3.20.0-py3-none-any.whl.metadata (2.1 kB)
Collecting typing-extensions>=4.10.0 (from torch)
  Using cached typing_extensions-4.15.0-py3-none-any.whl.metadata (3.3 kB)
Collecting setuptools (from torch)
  Using cached setuptools-80.9.0-py3-none-any.whl.metadata (6.6 kB)
Collecting sympy>=1.13.3 (from torch)
  Using cached sympy-1.14.0-py3-none-any.whl.metadata (12 kB)
Collecting networkx>=2.5.1 (from torch)
  Using cached networkx-3.5-py3-none-any.whl.metadata (6.3 kB)
Collecting jinja2 (from torch)
  Using cached jinja2-3.1.6-py3-none-any.whl.metadata (2.9 kB)
Collecting fsspec>=0.8.5 (from torch)
  Downloading fsspec-2025.10.0-py3-none-any.whl.metadata (10 kB)
Collecting nvidia-cuda-nvrtc-cu12==12.8.93 (from torch)
  Using cached nvidia_cuda_nvrtc_cu12-12.8.93-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl.metadata (1.7 kB)
Collecting nvidia-cuda-runtime-cu12==12.8.90 (from torch)
  Using cached nvidia_cuda_runtime_cu12-12.8.90-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.7 kB)
Collecting nvidia-cuda-cupti-cu12==12.8.90 (from torch)
  Using cached nvidia_cuda_cupti_cu12-12.8.90-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.7 kB)
Collecting nvidia-cudnn-cu12==9.10.2.21 (from torch)
  Using cached nvidia_cudnn_cu12-9.10.2.21-py3-none-manylinux_2_27_x86_64.whl.metadata (1.8 kB)
Collecting nvidia-cublas-cu12==12.8.4.1 (from torch)
  Using cached nvidia_cublas_cu12-12.8.4.1-py3-none-manylinux_2_27_x86_64.whl.metadata (1.7 kB)
Collecting nvidia-cufft-cu12==11.3.3.83 (from torch)
  Using cached nvidia_cufft_cu12-11.3.3.83-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.7 kB)
Collecting nvidia-curand-cu12==10.3.9.90 (from torch)
  Using cached nvidia_curand_cu12-10.3.9.90-py3-none-manylinux_2_27_x86_64.whl.metadata (1.7 kB)
Collecting nvidia-cusolver-cu12==11.7.3.90 (from torch)
  Using cached nvidia_cusolver_cu12-11.7.3.90-py3-none-manylinux_2_27_x86_64.whl.metadata (1.8 kB)
Collecting nvidia-cusparse-cu12==12.5.8.93 (from torch)
  Using cached nvidia_cusparse_cu12-12.5.8.93-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.8 kB)
Collecting nvidia-cusparselt-cu12==0.7.1 (from torch)
  Using cached nvidia_cusparselt_cu12-0.7.1-py3-none-manylinux2014_x86_64.whl.metadata (7.0 kB)
Collecting nvidia-nccl-cu12==2.27.5 (from torch)
  Using cached nvidia_nccl_cu12-2.27.5-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (2.0 kB)
Collecting nvidia-nvshmem-cu12==3.3.20 (from torch)
  Using cached nvidia_nvshmem_cu12-3.3.20-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (2.1 kB)
Collecting nvidia-nvtx-cu12==12.8.90 (from torch)
  Using cached nvidia_nvtx_cu12-12.8.90-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.8 kB)
Collecting nvidia-nvjitlink-cu12==12.8.93 (from torch)
  Using cached nvidia_nvjitlink_cu12-12.8.93-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl.metadata (1.7 kB)
Collecting nvidia-cufile-cu12==1.13.1.3 (from torch)
  Using cached nvidia_cufile_cu12-1.13.1.3-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.7 kB)
Collecting triton==3.5.0 (from torch)
  Using cached triton-3.5.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (1.7 kB)
Collecting huggingface-hub<1.0,>=0.34.0 (from transformers)
  Downloading huggingface_hub-0.36.0-py3-none-any.whl.metadata (14 kB)
Collecting numpy>=1.17 (from transformers)
  Using cached numpy-2.3.4-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (62 kB)
Collecting packaging>=20.0 (from transformers)
  Using cached packaging-25.0-py3-none-any.whl.metadata (3.3 kB)
Collecting pyyaml>=5.1 (from transformers)
  Using cached pyyaml-6.0.3-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (2.4 kB)
Collecting regex!=2019.12.17 (from transformers)
  Downloading regex-2025.10.23-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (40 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 40.5/40.5 kB 15.7 MB/s eta 0:00:00
Collecting requests (from transformers)
  Using cached requests-2.32.5-py3-none-any.whl.metadata (4.9 kB)
Collecting tokenizers<=0.23.0,>=0.22.0 (from transformers)
  Using cached tokenizers-0.22.1-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.8 kB)
Collecting safetensors>=0.4.3 (from transformers)
  Using cached safetensors-0.6.2-cp38-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (4.1 kB)
Collecting tqdm>=4.27 (from transformers)
  Using cached tqdm-4.67.1-py3-none-any.whl.metadata (57 kB)
Collecting hf-xet<2.0.0,>=1.1.3 (from huggingface-hub<1.0,>=0.34.0->transformers)
  Downloading hf_xet-1.2.0-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (4.9 kB)
Collecting mpmath<1.4,>=1.1.0 (from sympy>=1.13.3->torch)
  Using cached mpmath-1.3.0-py3-none-any.whl.metadata (8.6 kB)
Collecting MarkupSafe>=2.0 (from jinja2->torch)
  Using cached markupsafe-3.0.3-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (2.7 kB)
Collecting charset_normalizer<4,>=2 (from requests->transformers)
  Using cached charset_normalizer-3.4.4-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (37 kB)
Collecting idna<4,>=2.5 (from requests->transformers)
  Using cached idna-3.11-py3-none-any.whl.metadata (8.4 kB)
Collecting urllib3<3,>=1.21.1 (from requests->transformers)
  Using cached urllib3-2.5.0-py3-none-any.whl.metadata (6.5 kB)
Collecting certifi>=2017.4.17 (from requests->transformers)
  Using cached certifi-2025.10.5-py3-none-any.whl.metadata (2.5 kB)
Using cached torch-2.9.0-cp312-cp312-manylinux_2_28_x86_64.whl (899.7 MB)
Using cached nvidia_cublas_cu12-12.8.4.1-py3-none-manylinux_2_27_x86_64.whl (594.3 MB)
Using cached nvidia_cuda_cupti_cu12-12.8.90-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (10.2 MB)
Using cached nvidia_cuda_nvrtc_cu12-12.8.93-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl (88.0 MB)
Using cached nvidia_cuda_runtime_cu12-12.8.90-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (954 kB)
Using cached nvidia_cudnn_cu12-9.10.2.21-py3-none-manylinux_2_27_x86_64.whl (706.8 MB)
Using cached nvidia_cufft_cu12-11.3.3.83-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (193.1 MB)
Using cached nvidia_cufile_cu12-1.13.1.3-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (1.2 MB)
Using cached nvidia_curand_cu12-10.3.9.90-py3-none-manylinux_2_27_x86_64.whl (63.6 MB)
Using cached nvidia_cusolver_cu12-11.7.3.90-py3-none-manylinux_2_27_x86_64.whl (267.5 MB)
Using cached nvidia_cusparse_cu12-12.5.8.93-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (288.2 MB)
Using cached nvidia_cusparselt_cu12-0.7.1-py3-none-manylinux2014_x86_64.whl (287.2 MB)
Using cached nvidia_nccl_cu12-2.27.5-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (322.3 MB)
Using cached nvidia_nvjitlink_cu12-12.8.93-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl (39.3 MB)
Using cached nvidia_nvshmem_cu12-3.3.20-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (124.7 MB)
Using cached nvidia_nvtx_cu12-12.8.90-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (89 kB)
Using cached triton-3.5.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (170.5 MB)
Using cached transformers-4.57.1-py3-none-any.whl (12.0 MB)
Downloading fsspec-2025.10.0-py3-none-any.whl (200 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 201.0/201.0 kB 37.4 MB/s eta 0:00:00
Downloading huggingface_hub-0.36.0-py3-none-any.whl (566 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 566.1/566.1 kB 86.9 MB/s eta 0:00:00
Using cached networkx-3.5-py3-none-any.whl (2.0 MB)
Using cached numpy-2.3.4-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (16.6 MB)
Using cached packaging-25.0-py3-none-any.whl (66 kB)
Using cached pyyaml-6.0.3-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (807 kB)
Downloading regex-2025.10.23-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (803 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 803.4/803.4 kB 81.0 MB/s eta 0:00:00
Using cached safetensors-0.6.2-cp38-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (485 kB)
Using cached sympy-1.14.0-py3-none-any.whl (6.3 MB)
Using cached tokenizers-0.22.1-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (3.3 MB)
Using cached tqdm-4.67.1-py3-none-any.whl (78 kB)
Using cached typing_extensions-4.15.0-py3-none-any.whl (44 kB)
Using cached filelock-3.20.0-py3-none-any.whl (16 kB)
Using cached jinja2-3.1.6-py3-none-any.whl (134 kB)
Using cached requests-2.32.5-py3-none-any.whl (64 kB)
Using cached setuptools-80.9.0-py3-none-any.whl (1.2 MB)
Using cached certifi-2025.10.5-py3-none-any.whl (163 kB)
Using cached charset_normalizer-3.4.4-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (153 kB)
Downloading hf_xet-1.2.0-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (3.3 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.3/3.3 MB 103.8 MB/s eta 0:00:00
Using cached idna-3.11-py3-none-any.whl (71 kB)
Using cached markupsafe-3.0.3-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (22 kB)
Using cached mpmath-1.3.0-py3-none-any.whl (536 kB)
Using cached urllib3-2.5.0-py3-none-any.whl (129 kB)
Installing collected packages: nvidia-cusparselt-cu12, mpmath, urllib3, typing-extensions, triton, tqdm, sympy, setuptools, safetensors, regex, pyyaml, packaging, nvidia-nvtx-cu12, nvidia-nvshmem-cu12, nvidia-nvjitlink-cu12, nvidia-nccl-cu12, nvidia-curand-cu12, nvidia-cufile-cu12, nvidia-cuda-runtime-cu12, nvidia-cuda-nvrtc-cu12, nvidia-cuda-cupti-cu12, nvidia-cublas-cu12, numpy, networkx, MarkupSafe, idna, hf-xet, fsspec, filelock, charset_normalizer, certifi, requests, nvidia-cusparse-cu12, nvidia-cufft-cu12, nvidia-cudnn-cu12, jinja2, nvidia-cusolver-cu12, huggingface-hub, torch, tokenizers, transformers
Successfully installed MarkupSafe-3.0.3 certifi-2025.10.5 charset_normalizer-3.4.4 filelock-3.20.0 fsspec-2025.10.0 hf-xet-1.2.0 huggingface-hub-0.36.0 idna-3.11 jinja2-3.1.6 mpmath-1.3.0 networkx-3.5 numpy-2.3.4 nvidia-cublas-cu12-12.8.4.1 nvidia-cuda-cupti-cu12-12.8.90 nvidia-cuda-nvrtc-cu12-12.8.93 nvidia-cuda-runtime-cu12-12.8.90 nvidia-cudnn-cu12-9.10.2.21 nvidia-cufft-cu12-11.3.3.83 nvidia-cufile-cu12-1.13.1.3 nvidia-curand-cu12-10.3.9.90 nvidia-cusolver-cu12-11.7.3.90 nvidia-cusparse-cu12-12.5.8.93 nvidia-cusparselt-cu12-0.7.1 nvidia-nccl-cu12-2.27.5 nvidia-nvjitlink-cu12-12.8.93 nvidia-nvshmem-cu12-3.3.20 nvidia-nvtx-cu12-12.8.90 packaging-25.0 pyyaml-6.0.3 regex-2025.10.23 requests-2.32.5 safetensors-0.6.2 setuptools-80.9.0 sympy-1.14.0 tokenizers-0.22.1 torch-2.9.0 tqdm-4.67.1 transformers-4.57.1 triton-3.5.0 typing-extensions-4.15.0 urllib3-2.5.0
```

### `pick_imp.py`
このプログラムは，入力された文章から「命令・依頼」に該当する文を自動で抽出します．
モデル出力に基づいて，文ごとに「命令・依頼」「説明」「質問」「状況」のスコアを算出し，
しきい値を超えた文をレポートとして anser/ に保存します．

抽出の挙動は以下のオプションで制御できます．

「命令・依頼」とみなすスコア閾値（デフォルト：0.5）
`--threshold`
閾値を超えた段落を全て出力（none / paragraph）
`--chunk`

- **入力**は、dataに入った文章．
- **出力**は、「命令・依頼」がスコアを上回る文，そのスコア．

**使ったdataファイル**
```
課題 2-1 についての相談ですね
あなたの作成した phpMyAdminのvalues.yaml を改めて確認してください．
70行目から90行目のimageの箇所をよく確認してください
626行目から759行目のmetricsの箇所をよく確認してください

ファイルのMariaDBのvalues.yamlがないので，ファイルを作成してください．
```

**使い方**
```
(venv) arita@sre000h:~/pick_imp$ python3 pick_imp.py
Device set to use cpu
Wrote: /home/arita/pick_imp/anser/answer_20251103_050543.txt
(venv) arita@sre000h:~/pick_imp$ 
```

**実行結果**
2~5行目が「命令・依頼」を意味する文として出てきた．
```
(venv) arita@sre000h:~/pick_imp$ cat /home/arita/pick_imp/anser/answer_20251103_050543.txt
# 命令/依頼 抽出レポート (最小版)  (20251103_050543 JST)
data path: /home/arita/pick_imp/data
model: MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7
total_extracted_sentences: 4
labels: ['命令・依頼', '説明', '質問', '状況']
threshold: 0.5
chunk_mode: none

================================================================================
FILE: /home/arita/pick_imp/data
EXTRACTED: 4
--------------------------------------------------------------------------------
[line    2] あなたの作成した phpMyAdminのvalues.yaml を改めて確認してください
  -> label=命令・依頼  score=0.511
[line    3] 70行目から90行目のimageの箇所をよく確認してください
  -> label=命令・依頼  score=0.54
[line    4] 626行目から759行目のmetricsの箇所をよく確認してください
  -> label=命令・依頼(fallback)  score=0.6
[line    5] ファイルのMariaDBのvalues.yamlがないので，ファイルを作成してください
  -> label=命令・依頼(fallback)  score=0.6
--------------------------------------------------------------------------------
【Original Text】
課題 2-1 についての相談ですね
あなたの作成した phpMyAdminのvalues.yaml を改めて確認してください．
70行目から90行目のimageの箇所をよく確認してください
626行目から759行目のmetricsの箇所をよく確認してください

ファイルのMariaDBのvalues.yamlがないので，ファイルを作成してください．
(venv) arita@sre000h:~/pick_imp$
```







