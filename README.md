**Este Notebook foi usado para uma demonstração interna com o time do AI Studio para mostrar um fluxo de trabalho comum em DS 🤗**

# Let's bora? Explicações iniciais

### O que é um classificador de Spam? 🤔

Um classificador de spam é como um detetive digital esperto que examina suas mensagens e decide quais são boas e quais são "lixo". Imagine que ele tem uma lupa mágica que pode ver padrões suspeitos nas palavras, como "dinheiro fácil" ou "clique aqui". Quando ele encontra essas pistas, coloca a mensagem na "lixeira" de spam, protegendo sua caixa de entrada de ser invadida por ofertas de príncipes nigerianos ou de produtos milagrosos que ninguém pediu!

## Sim, o HuggingFace é uma queen! 😍

### Da onde vem nosso Dataset? 🤔

Vem da biblioteca Datasets, também do HuggingFace! É uma ferramenta para facilitar o acesso e o uso de conjuntos de dados em projetos de machine learning. Ela permite carregar, processar e manipular dados, suportando uma ampla variedade de formatos e fontes de dados.

### Quem é Transformers? 🤔

A biblioteca Transformers do HuggingFace é uma ferramenta para trabalhar com modelos de linguagem natural (NLP) baseados na arquitetura Transformers, como BERT, GPT, e T5. Ela fornece uma interface fácil de usar para carregar, treinar e utilizar esses modelos para tarefas como tradução, resumo de texto, classificação e geração de texto!

### Por que precisamos do Accelerate se não importamos ele? 🤔

O Accelerate é uma biblioteca TAMBÉM da Hugging Face que sereve para facilitar o uso de múltiplos dispositivos (como GPUs e TPUs) ao treinar e inferir modelos de aprendizado profundo. Mesmo que você não importe diretamente no seu código, você pode estar utilizando funcionalidades ou bibliotecas que dependem do Accelerate para melhorar a eficiência e o desempenho.

## BERT 🤖

BERT (Bidirectional Encoder Representations from Transformers) é um modelo de linguagem natural desenvolvido pelo Google que revolucionou o processamento de linguagem natural (NLP) ao entender o contexto de palavras em uma frase de forma bidirecional, ou seja, analisando as palavras à esquerda e à direita de cada palavra-alvo ao mesmo tempo.

O `bert-base-uncased` é uma versão do modelo BERT com 12 camadas (ou "transformers") e 110 milhões de parâmetros. Esses parâmetros são como "neurônios" em uma rede neural que aprendem a representar o significado das palavras.

# Mãos na massa (ou no AIS? 🤔)

## Imports


```python
!pip install transformers accelerate datasets wordcloud
```

    Collecting transformers
      Downloading transformers-4.42.4-py3-none-any.whl.metadata (43 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m43.6/43.6 kB[0m [31m575.1 kB/s[0m eta [36m0:00:00[0m [36m0:00:01[0m
    [?25hCollecting accelerate
      Downloading accelerate-0.32.1-py3-none-any.whl.metadata (18 kB)
    Collecting datasets
      Downloading datasets-2.20.0-py3-none-any.whl.metadata (19 kB)
    Collecting wordcloud
      Downloading wordcloud-1.9.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (3.4 kB)
    Requirement already satisfied: filelock in /opt/conda/lib/python3.10/site-packages (from transformers) (3.13.1)
    Collecting huggingface-hub<1.0,>=0.23.2 (from transformers)
      Downloading huggingface_hub-0.23.4-py3-none-any.whl.metadata (12 kB)
    Requirement already satisfied: numpy<2.0,>=1.17 in /opt/conda/lib/python3.10/site-packages (from transformers) (1.24.3)
    Requirement already satisfied: packaging>=20.0 in /opt/conda/lib/python3.10/site-packages (from transformers) (23.2)
    Requirement already satisfied: pyyaml>=5.1 in /opt/conda/lib/python3.10/site-packages (from transformers) (6.0.1)
    Collecting regex!=2019.12.17 (from transformers)
      Downloading regex-2024.5.15-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (40 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m40.9/40.9 kB[0m [31m1.5 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: requests in /opt/conda/lib/python3.10/site-packages (from transformers) (2.31.0)
    Collecting safetensors>=0.4.1 (from transformers)
      Downloading safetensors-0.4.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (3.8 kB)
    Collecting tokenizers<0.20,>=0.19 (from transformers)
      Downloading tokenizers-0.19.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.7 kB)
    Requirement already satisfied: tqdm>=4.27 in /opt/conda/lib/python3.10/site-packages (from transformers) (4.65.0)
    Requirement already satisfied: psutil in /opt/conda/lib/python3.10/site-packages (from accelerate) (5.9.8)
    Requirement already satisfied: torch>=1.10.0 in /opt/conda/lib/python3.10/site-packages (from accelerate) (2.0.0+cu118)
    Collecting pyarrow>=15.0.0 (from datasets)
      Downloading pyarrow-16.1.0-cp310-cp310-manylinux_2_28_x86_64.whl.metadata (3.0 kB)
    Collecting pyarrow-hotfix (from datasets)
      Downloading pyarrow_hotfix-0.6-py3-none-any.whl.metadata (3.6 kB)
    Requirement already satisfied: dill<0.3.9,>=0.3.0 in /opt/conda/lib/python3.10/site-packages (from datasets) (0.3.7)
    Requirement already satisfied: pandas in /opt/conda/lib/python3.10/site-packages (from datasets) (2.2.0)
    Collecting requests (from transformers)
      Downloading requests-2.32.3-py3-none-any.whl.metadata (4.6 kB)
    Collecting tqdm>=4.27 (from transformers)
      Downloading tqdm-4.66.4-py3-none-any.whl.metadata (57 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m57.6/57.6 kB[0m [31m1.3 MB/s[0m eta [36m0:00:00[0mta [36m0:00:01[0m
    [?25hCollecting xxhash (from datasets)
      Downloading xxhash-3.4.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (12 kB)
    Collecting multiprocess (from datasets)
      Downloading multiprocess-0.70.16-py310-none-any.whl.metadata (7.2 kB)
    Collecting fsspec<=2024.5.0,>=2023.1.0 (from fsspec[http]<=2024.5.0,>=2023.1.0->datasets)
      Downloading fsspec-2024.5.0-py3-none-any.whl.metadata (11 kB)
    Collecting aiohttp (from datasets)
      Downloading aiohttp-3.9.5-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (7.5 kB)
    Requirement already satisfied: pillow in /opt/conda/lib/python3.10/site-packages (from wordcloud) (10.2.0)
    Requirement already satisfied: matplotlib in /opt/conda/lib/python3.10/site-packages (from wordcloud) (3.8.2)
    Collecting aiosignal>=1.1.2 (from aiohttp->datasets)
      Downloading aiosignal-1.3.1-py3-none-any.whl.metadata (4.0 kB)
    Requirement already satisfied: attrs>=17.3.0 in /opt/conda/lib/python3.10/site-packages (from aiohttp->datasets) (23.2.0)
    Collecting frozenlist>=1.1.1 (from aiohttp->datasets)
      Downloading frozenlist-1.4.1-cp310-cp310-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (12 kB)
    Collecting multidict<7.0,>=4.5 (from aiohttp->datasets)
      Downloading multidict-6.0.5-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (4.2 kB)
    Collecting yarl<2.0,>=1.0 (from aiohttp->datasets)
      Downloading yarl-1.9.4-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (31 kB)
    Collecting async-timeout<5.0,>=4.0 (from aiohttp->datasets)
      Downloading async_timeout-4.0.3-py3-none-any.whl.metadata (4.2 kB)
    Requirement already satisfied: typing-extensions>=3.7.4.3 in /opt/conda/lib/python3.10/site-packages (from huggingface-hub<1.0,>=0.23.2->transformers) (4.12.1)
    Requirement already satisfied: charset-normalizer<4,>=2 in /opt/conda/lib/python3.10/site-packages (from requests->transformers) (2.0.4)
    Requirement already satisfied: idna<4,>=2.5 in /opt/conda/lib/python3.10/site-packages (from requests->transformers) (3.7)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /opt/conda/lib/python3.10/site-packages (from requests->transformers) (2.2.1)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/lib/python3.10/site-packages (from requests->transformers) (2023.7.22)
    Requirement already satisfied: sympy in /opt/conda/lib/python3.10/site-packages (from torch>=1.10.0->accelerate) (1.12)
    Requirement already satisfied: networkx in /opt/conda/lib/python3.10/site-packages (from torch>=1.10.0->accelerate) (3.3)
    Requirement already satisfied: jinja2 in /opt/conda/lib/python3.10/site-packages (from torch>=1.10.0->accelerate) (3.1.4)
    Requirement already satisfied: triton==2.0.0 in /opt/conda/lib/python3.10/site-packages (from torch>=1.10.0->accelerate) (2.0.0)
    Requirement already satisfied: cmake in /opt/conda/lib/python3.10/site-packages (from triton==2.0.0->torch>=1.10.0->accelerate) (3.25.0)
    Requirement already satisfied: lit in /opt/conda/lib/python3.10/site-packages (from triton==2.0.0->torch>=1.10.0->accelerate) (15.0.7)
    Requirement already satisfied: contourpy>=1.0.1 in /opt/conda/lib/python3.10/site-packages (from matplotlib->wordcloud) (1.2.1)
    Requirement already satisfied: cycler>=0.10 in /opt/conda/lib/python3.10/site-packages (from matplotlib->wordcloud) (0.12.1)
    Requirement already satisfied: fonttools>=4.22.0 in /opt/conda/lib/python3.10/site-packages (from matplotlib->wordcloud) (4.51.0)
    Requirement already satisfied: kiwisolver>=1.3.1 in /opt/conda/lib/python3.10/site-packages (from matplotlib->wordcloud) (1.4.5)
    Requirement already satisfied: pyparsing>=2.3.1 in /opt/conda/lib/python3.10/site-packages (from matplotlib->wordcloud) (3.1.2)
    Requirement already satisfied: python-dateutil>=2.7 in /opt/conda/lib/python3.10/site-packages (from matplotlib->wordcloud) (2.9.0)
    Collecting dill<0.3.9,>=0.3.0 (from datasets)
      Downloading dill-0.3.8-py3-none-any.whl.metadata (10 kB)
    Requirement already satisfied: pytz>=2020.1 in /opt/conda/lib/python3.10/site-packages (from pandas->datasets) (2023.4)
    Requirement already satisfied: tzdata>=2022.7 in /opt/conda/lib/python3.10/site-packages (from pandas->datasets) (2024.1)
    Requirement already satisfied: six>=1.5 in /opt/conda/lib/python3.10/site-packages (from python-dateutil>=2.7->matplotlib->wordcloud) (1.16.0)
    Requirement already satisfied: MarkupSafe>=2.0 in /opt/conda/lib/python3.10/site-packages (from jinja2->torch>=1.10.0->accelerate) (2.1.5)
    Requirement already satisfied: mpmath>=0.19 in /opt/conda/lib/python3.10/site-packages (from sympy->torch>=1.10.0->accelerate) (1.3.0)
    Downloading transformers-4.42.4-py3-none-any.whl (9.3 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m9.3/9.3 MB[0m [31m25.0 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hDownloading accelerate-0.32.1-py3-none-any.whl (314 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m314.1/314.1 kB[0m [31m5.7 MB/s[0m eta [36m0:00:00[0m:00:01[0m
    [?25hDownloading datasets-2.20.0-py3-none-any.whl (547 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m547.8/547.8 kB[0m [31m14.5 MB/s[0m eta [36m0:00:00[0m00:01[0m
    [?25hDownloading wordcloud-1.9.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (511 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m511.1/511.1 kB[0m [31m19.0 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading fsspec-2024.5.0-py3-none-any.whl (316 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m316.1/316.1 kB[0m [31m13.1 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading aiohttp-3.9.5-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.2 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.2/1.2 MB[0m [31m34.1 MB/s[0m eta [36m0:00:00[0m00:01[0m
    [?25hDownloading huggingface_hub-0.23.4-py3-none-any.whl (402 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m402.6/402.6 kB[0m [31m12.1 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading pyarrow-16.1.0-cp310-cp310-manylinux_2_28_x86_64.whl (40.8 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m40.8/40.8 MB[0m [31m23.9 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hDownloading regex-2024.5.15-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (775 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m775.1/775.1 kB[0m [31m17.7 MB/s[0m eta [36m0:00:00[0m00:01[0m
    [?25hDownloading requests-2.32.3-py3-none-any.whl (64 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m64.9/64.9 kB[0m [31m3.2 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading safetensors-0.4.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.2 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.2/1.2 MB[0m [31m26.9 MB/s[0m eta [36m0:00:00[0m:00:01[0m
    [?25hDownloading tokenizers-0.19.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (3.6 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m3.6/3.6 MB[0m [31m37.2 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hDownloading tqdm-4.66.4-py3-none-any.whl (78 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m78.3/78.3 kB[0m [31m5.3 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading multiprocess-0.70.16-py310-none-any.whl (134 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m134.8/134.8 kB[0m [31m12.7 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading dill-0.3.8-py3-none-any.whl (116 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m116.3/116.3 kB[0m [31m3.9 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading pyarrow_hotfix-0.6-py3-none-any.whl (7.9 kB)
    Downloading xxhash-3.4.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (194 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m194.1/194.1 kB[0m [31m8.1 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading aiosignal-1.3.1-py3-none-any.whl (7.6 kB)
    Downloading async_timeout-4.0.3-py3-none-any.whl (5.7 kB)
    Downloading frozenlist-1.4.1-cp310-cp310-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (239 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m239.5/239.5 kB[0m [31m5.4 MB/s[0m eta [36m0:00:00[0m:00:01[0m
    [?25hDownloading multidict-6.0.5-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (124 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m124.3/124.3 kB[0m [31m8.3 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading yarl-1.9.4-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (301 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m301.6/301.6 kB[0m [31m16.4 MB/s[0m eta [36m0:00:00[0m
    [?25hInstalling collected packages: xxhash, tqdm, safetensors, requests, regex, pyarrow-hotfix, pyarrow, multidict, fsspec, frozenlist, dill, async-timeout, yarl, multiprocess, huggingface-hub, aiosignal, wordcloud, tokenizers, aiohttp, transformers, datasets, accelerate
      Attempting uninstall: tqdm
        Found existing installation: tqdm 4.65.0
        Uninstalling tqdm-4.65.0:
          Successfully uninstalled tqdm-4.65.0
      Attempting uninstall: requests
        Found existing installation: requests 2.31.0
        Uninstalling requests-2.31.0:
          Successfully uninstalled requests-2.31.0
      Attempting uninstall: pyarrow
        Found existing installation: pyarrow 12.0.1
        Uninstalling pyarrow-12.0.1:
          Successfully uninstalled pyarrow-12.0.1
      Attempting uninstall: dill
        Found existing installation: dill 0.3.7
        Uninstalling dill-0.3.7:
          Successfully uninstalled dill-0.3.7
    [31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    mlflow 2.6.0 requires pyarrow<13,>=4.0.0, but you have pyarrow 16.1.0 which is incompatible.
    tensorflow 2.13.0 requires typing-extensions<4.6.0,>=3.6.6, but you have typing-extensions 4.12.1 which is incompatible.[0m[31m
    [0mSuccessfully installed accelerate-0.32.1 aiohttp-3.9.5 aiosignal-1.3.1 async-timeout-4.0.3 datasets-2.20.0 dill-0.3.8 frozenlist-1.4.1 fsspec-2024.5.0 huggingface-hub-0.23.4 multidict-6.0.5 multiprocess-0.70.16 pyarrow-16.1.0 pyarrow-hotfix-0.6 regex-2024.5.15 requests-2.32.3 safetensors-0.4.3 tokenizers-0.19.1 tqdm-4.66.4 transformers-4.42.4 wordcloud-1.9.3 xxhash-3.4.1 yarl-1.9.4
    


```python
from datasets import load_dataset, load_metric, DatasetDict
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
from sklearn.model_selection import train_test_split
import mlflow
from mlflow.types.schema import Schema, ColSpec
from mlflow.types import ParamSchema, ParamSpec
from mlflow.models import ModelSignature
import torch
import numpy as np
```

## [SMS SPAM](https://huggingface.co/datasets/ucirvine/sms_spam)


```python
dataset = load_dataset("sms_spam", trust_remote_code=True)
```


    Downloading builder script:   0%|          | 0.00/3.21k [00:00<?, ?B/s]



    Downloading readme:   0%|          | 0.00/4.87k [00:00<?, ?B/s]



    Downloading data: 0.00B [00:00, ?B/s]



    Generating train split:   0%|          | 0/5574 [00:00<?, ? examples/s]



```python
for _, data in dataset.items():
    print(data.info.description)
```

    The SMS Spam Collection v.1 is a public set of SMS labeled messages that have been collected for mobile phone spam research.
    It has one collection composed by 5,574 English, real and non-enconded messages, tagged according being legitimate (ham) or spam.
    
    


```python
dataset
```




    DatasetDict({
        train: Dataset({
            features: ['sms', 'label'],
            num_rows: 5574
        })
    })



## Análise Exploratória de Dados (EDA) 📊


```python
df = dataset['train'].to_pandas()
df.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sms</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Go until jurong point, crazy.. Available only ...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ok lar... Joking wif u oni...\n</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>U dun say so early hor... U c already then say...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Nah I don't think he goes to usf, he lives aro...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>FreeMsg Hey there darling it's been 3 week's n...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Even my brother is not like to speak with me. ...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>As per your request 'Melle Melle (Oru Minnamin...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>WINNER!! As a valued network customer you have...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Had your mobile 11 months or more? U R entitle...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



### Checando balanceamento


```python
plt.figure(figsize=(6, 4))
sns.countplot(x='label', data=df)
plt.title('Class Distribution')
plt.xlabel('Label')
plt.ylabel('Count')
plt.show()
```


    
![png](E2E-com-BERT%20%281%29_files/E2E-com-BERT%20%281%29_23_0.png)
    


### Tamanho das mensagens


```python
df['length'] = df['sms'].apply(len)
df.sample(1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sms</th>
      <th>label</th>
      <th>length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>401</th>
      <td>FREE RINGTONE text FIRST to 87131 for a poly o...</td>
      <td>1</td>
      <td>157</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.histplot(df[df['label'] == 0]['length'], bins=50, color='blue', kde=True)
plt.title('Ham Message Length Distribution')
plt.xlabel('Message Length')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
sns.histplot(df[df['label'] == 1]['length'], bins=50, color='red', kde=True)
plt.title('Spam Message Length Distribution')
plt.xlabel('Message Length')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
```

    /opt/conda/lib/python3.10/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    /opt/conda/lib/python3.10/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    


    
![png](E2E-com-BERT%20%281%29_files/E2E-com-BERT%20%281%29_26_1.png)
    


### Nuvem de Palavras com a WordCloud 😶‍🌫️


```python
spam_words = ' '.join(list(df[df['label'] == 1]['sms']))
ham_words = ' '.join(list(df[df['label'] == 0]['sms']))
```


```python
spam_wordcloud = WordCloud(width=600, height=400, background_color='black').generate(spam_words)
ham_wordcloud = WordCloud(width=600, height=400, background_color='white').generate(ham_words)
```


```python
plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.imshow(spam_wordcloud, interpolation='bilinear')
plt.title('Spam Word Cloud')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(ham_wordcloud, interpolation='bilinear')
plt.title('Ham Word Cloud')
plt.axis('off')

plt.show()
```


    
![png](E2E-com-BERT%20%281%29_files/E2E-com-BERT%20%281%29_30_0.png)
    


## Balanceando o Dataset


```python
df_spam = df[df['label'] == 1]
df_not_spam = df[df['label'] == 0]
```


```python
spam_count = len(df_spam)
not_spam_count = len(df_not_spam)

print(f'SPAM: {spam_count} | HAM: {not_spam_count}')
```

    SPAM: 747 | HAM: 4827
    


```python
df_spam_oversampled = df_spam.sample(not_spam_count, replace=True, random_state=42)
```


```python
df_balanced = pd.concat([df_spam_oversampled, df_not_spam]).sample(frac=1, random_state=42).reset_index(drop=True)
```


```python
balanced_dataset = dataset['train'].from_pandas(df_balanced)
```


```python
print("Number of spam messages:", sum(balanced_dataset['label']))
print("Number of not spam messages:", len(balanced_dataset['label']) - sum(balanced_dataset['label']))
```

    Number of spam messages: 4827
    Number of not spam messages: 4827
    

## Divisão de Treino e Teste


```python
dataset = dataset['train'].train_test_split(test_size=0.2)
dataset
```




    DatasetDict({
        train: Dataset({
            features: ['sms', 'label'],
            num_rows: 4459
        })
        test: Dataset({
            features: ['sms', 'label'],
            num_rows: 1115
        })
    })



## Tokenização 🤗

Tokenização é o processo de dividir um texto em unidades menores chamadas tokens, que podem ser palavras, subpalavras ou caracteres. Por exemplo, a frase "Eu gosto de maçãs" pode ser tokenizada em ["Eu", "gosto", "de", "maçãs"].

🤖 Modelos de linguagem natural não podem processar texto bruto diretamente. Eles precisam que o texto seja convertido em uma forma numérica!


```python
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', trust_remote_code=True)
```


    tokenizer_config.json:   0%|          | 0.00/48.0 [00:00<?, ?B/s]



    vocab.txt:   0%|          | 0.00/232k [00:00<?, ?B/s]



    tokenizer.json:   0%|          | 0.00/466k [00:00<?, ?B/s]



    config.json:   0%|          | 0.00/570 [00:00<?, ?B/s]



```python
def tokenize_function(data):
    return tokenizer(data['sms'], padding="max_length", truncation=True)
```


```python
tokenized_datasets = dataset.map(tokenize_function, batched=True)
```


    Map:   0%|          | 0/4459 [00:00<?, ? examples/s]



    Map:   0%|          | 0/1115 [00:00<?, ? examples/s]



```python
tokenized_datasets
```




    DatasetDict({
        train: Dataset({
            features: ['sms', 'label', 'input_ids', 'token_type_ids', 'attention_mask'],
            num_rows: 4459
        })
        test: Dataset({
            features: ['sms', 'label', 'input_ids', 'token_type_ids', 'attention_mask'],
            num_rows: 1115
        })
    })




```python
tokenized_datasets = tokenized_datasets.remove_columns(["sms"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")
```

## Hora de Treinar! 🤖

### Carregando o modelo


```python
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2, trust_remote_code=True)
```

    The argument `trust_remote_code` is to be used with Auto classes. It has no effect here and is ignored.
    


    model.safetensors:   0%|          | 0.00/440M [00:00<?, ?B/s]


    Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    

### Argumentos de Treino


```python
training_args = TrainingArguments(
    # Diretório onde os modelos e checkpoints serão salvos
    output_dir="./results",
    # Estratégia de avaliação: 'epoch' significa que a avaliação será feita ao final de cada época
    evaluation_strategy="epoch",
    # Taxa de aprendizado inicial para o otimizador
    learning_rate=2e-5,
    # Tamanho do lote de treinamento por dispositivo (GPU/CPU)
    per_device_train_batch_size=8,
    # Tamanho do lote de avaliação por dispositivo (GPU/CPU)
    per_device_eval_batch_size=8,
    # Número de épocas (passes completos pelo conjunto de dados de treinamento)
    num_train_epochs=2,
    # Taxa de decaimento do peso para regularização (evita overfitting)
    weight_decay=0.01,
    # Relatar o progresso do treinamento para MLflow (ferramenta de acompanhamento de experimentos)
    report_to='mlflow',
    # Quantos passos de treinamento entre cada salvamento de modelo
    save_steps=300
)
```

    /opt/conda/lib/python3.10/site-packages/transformers/training_args.py:1494: FutureWarning: `evaluation_strategy` is deprecated and will be removed in version 4.46 of 🤗 Transformers. Use `eval_strategy` instead
      warnings.warn(
    

### Criando uma métrica de acurácia


```python
def compute_metrics(eval_pred):
    # Desempacotando a tupla eval_pred em logits e labels
    logits, labels = eval_pred
    # Calculando as previsões ao longo do último eixo
    # np.argmax retorna os índices dos maiores valores ao longo do eixo especificado
    # obtendo as classes previstas
    predictions = np.argmax(logits, axis=-1)
    # Carregando a métrica de acurácia usando a função load_metric do HuggingFace
    # trust_remote_code=True permite carregar métricas de código remoto confiável
    metric = load_metric("accuracy", trust_remote_code=True)
    # Computando a métrica de acurácia comparando as previsões com os rótulos verdadeiros
    # metric.compute calcula a acurácia usando as previsões e as referências (rótulos verdadeiros)
    return metric.compute(predictions=predictions, references=labels)
```

### Setting Trainer


```python
def sample_dataset(dataset, sample_size=10, seed=42):
    np.random.seed(seed)
    indices = np.random.choice(len(dataset), size=sample_size, replace=False)
    return dataset.select(indices)


sampled_train_dataset = sample_dataset(tokenized_datasets["train"], sample_size=10)
sampled_eval_dataset = sample_dataset(tokenized_datasets["test"], sample_size=10)
```


```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=sampled_train_dataset,
    eval_dataset=sampled_eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
```


```python
mlflow.set_experiment('BERT for Spam')
```

    2024/07/12 18:54:02 INFO mlflow.tracking.fluent: Experiment with name 'BERT for Spam' does not exist. Creating a new experiment.
    




    <Experiment: artifact_location='/phoenix/mlflow/561613461314365749', creation_time=1720810442113, experiment_id='561613461314365749', last_update_time=1720810442113, lifecycle_stage='active', name='BERT for Spam', tags={}>




```python
trainer.train()
```



    <div>

      <progress value='4' max='4' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [4/4 01:21, Epoch 2/2]
    </div>
    <table border="1" class="dataframe">
  <thead>
 <tr style="text-align: left;">
      <th>Epoch</th>
      <th>Training Loss</th>
      <th>Validation Loss</th>
      <th>Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>No log</td>
      <td>0.684168</td>
      <td>0.700000</td>
    </tr>
    <tr>
      <td>2</td>
      <td>No log</td>
      <td>0.704892</td>
      <td>0.500000</td>
    </tr>
  </tbody>
</table><p>


    /tmp/ipykernel_148/3706206025.py:10: FutureWarning: load_metric is deprecated and will be removed in the next major version of datasets. Use 'evaluate.load' instead, from the new library 🤗 Evaluate: https://huggingface.co/docs/evaluate
      metric = load_metric("accuracy", trust_remote_code=True)
    


    Downloading builder script:   0%|          | 0.00/1.65k [00:00<?, ?B/s]





    TrainOutput(global_step=4, training_loss=0.6539950370788574, metrics={'train_runtime': 109.4954, 'train_samples_per_second': 0.183, 'train_steps_per_second': 0.037, 'total_flos': 5262221107200.0, 'train_loss': 0.6539950370788574, 'epoch': 2.0})



### Saving the model


```python
model.save_pretrained("./spam-detection-model")
tokenizer.save_pretrained("./spam-detection-model")
```




    ('./spam-detection-model/tokenizer_config.json',
     './spam-detection-model/special_tokens_map.json',
     './spam-detection-model/vocab.txt',
     './spam-detection-model/added_tokens.json')



### Colocando na HuggingFace


```python
model = AutoModelForSequenceClassification.from_pretrained("./spam-detection-model")
tokenizer = AutoTokenizer.from_pretrained("./spam-detection-model")
```


```python
repo_name = "E2E-BERT-FOR-SPAM"
username = "morgana-rodrigues"
full_repo_name = f"{username}/{repo_name}"
full_repo_name
```




    'morgana-rodrigues/E2E-BERT-FOR-SPAM'




```python
# create_repo(full_repo_name, exist_ok=True, token='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')
```

## Time to Deploy 😎

### Crindo o pyfunc Model



Essa classe é utilizada para definir e encapsular um modelo Python dentro do MLflow com a lógica de preprocessamento e postprocessamento dos dados!


```python
class BERTSPAM(mlflow.pyfunc.PythonModel):
    def _preprocess(self, inputs):
        email = inputs['email'][0]
        print("pre processing", email)
        return email
        
    def load_context(self, context):
        self.model = BertForSequenceClassification.from_pretrained(context.artifacts["model"])
        self.tokenizer = BertTokenizer.from_pretrained(context.artifacts["tokenizer"])
        
    def predict(self, context, model_input):
        text = self._preprocess(model_input)
        print("TEXT", text)
        inputs = self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
        inputs = {key: value for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predicted_class_id = logits.argmax().item()

        if predicted_class_id == 1:
            return "Spam"
        else:
            return "Not Spam"

    @classmethod
    def log_model(cls, model_name, model, tokenizer): #eg (model, '', 'my_model')
        input_schema = Schema(
            [
                ColSpec("string", "email"),
            ]
        )
        output_schema = Schema(
            [
                ColSpec("string", "predicted_class_id")
            ]
        )
              
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)

        model.save_pretrained(model_name)
        tokenizer.save_pretrained(model_name)
             
        requirements = [
            "transformers==4.37.0",
            "mlflow==2.6.0",
            "numpy==1.24.3",
            "torch==2.0.0",
            "tqdm==4.65.0",
        ]
        mlflow.pyfunc.log_model(
            model_name,
            python_model=cls(),
            artifacts={"model": model_name, 'tokenizer': model_name},
            signature=signature,
            pip_requirements=requirements
        )
```


```python
mlflow.set_experiment(experiment_name='BERT Deploy')
```

    2024/07/12 19:04:04 INFO mlflow.tracking.fluent: Experiment with name 'BERT Deploy' does not exist. Creating a new experiment.
    




    <Experiment: artifact_location='/phoenix/mlflow/255326068679959782', creation_time=1720811044383, experiment_id='255326068679959782', last_update_time=1720811044383, lifecycle_stage='active', name='BERT Deploy', tags={}>




```python
tokenizer = AutoTokenizer.from_pretrained("morgana-rodrigues/bert-for-spam")
model = AutoModelForSequenceClassification.from_pretrained("morgana-rodrigues/bert-for-spam")
```


    tokenizer_config.json:   0%|          | 0.00/1.30k [00:00<?, ?B/s]



    vocab.txt:   0%|          | 0.00/262k [00:00<?, ?B/s]



    special_tokens_map.json:   0%|          | 0.00/132 [00:00<?, ?B/s]



    config.json:   0%|          | 0.00/754 [00:00<?, ?B/s]



    model.safetensors:   0%|          | 0.00/438M [00:00<?, ?B/s]



```python
with mlflow.start_run(run_name='BERTSPAM') as run:
    print(f"Run's Artifact URI: {run.info.artifact_uri}")
    BERTSPAM.log_model(model_name='BERTSPAM', model=model, tokenizer=tokenizer)
    mlflow.register_model(model_uri = f"runs:/{run.info.run_id}/BERTSPAM", name='BERTSPAM')
```

    Run's Artifact URI: /phoenix/mlflow/255326068679959782/7383069d98db4a91bc63c65f9e104eb8/artifacts
    


    Downloading artifacts:   0%|          | 0/6 [00:00<?, ?it/s]



    Downloading artifacts:   0%|          | 0/6 [00:00<?, ?it/s]


    /opt/conda/lib/python3.10/site-packages/_distutils_hack/__init__.py:26: UserWarning: Setuptools is replacing distutils.
      warnings.warn("Setuptools is replacing distutils.")
    Successfully registered model 'BERTSPAM'.
    2024/07/12 19:05:37 INFO mlflow.tracking._model_registry.client: Waiting up to 300 seconds for model version to finish creation. Model name: BERTSPAM, version 1
    Created version '1' of model 'BERTSPAM'.
    

### Testando 🤓


```python
client = mlflow.MlflowClient()
model_metadata = client.get_latest_versions("BERTSPAM", stages=["None"])
latest_model_version = model_metadata[0].version
print(latest_model_version, mlflow.models.get_model_info(f"models:/BERTSPAM/{latest_model_version}").signature)
```

    1 inputs: 
      ['email': string]
    outputs: 
      ['predicted_class_id': string]
    params: 
      None
    
    


```python
model = mlflow.pyfunc.load_model(model_uri=f"models:/BERTSPAM/{latest_model_version}")
```

    2024/07/12 19:06:28 WARNING mlflow.pyfunc: Detected one or more mismatches between the model's dependencies and the current Python environment:
     - transformers (current: 4.42.4, required: transformers==4.37.0)
     - tqdm (current: 4.66.4, required: tqdm==4.65.0)
    To fix the mismatches, call `mlflow.pyfunc.get_model_dependencies(model_uri)` to fetch the model's environment and install dependencies using the resulting environment file.
    


```python
email = "Congratulations!!! You've won a $1000 Walmart gift card!!! Click here to claim your prize."
```


```python
model.predict({"email": email})
```

    pre processing Congratulations!!! You've won a $1000 Walmart gift card!!! Click here to claim your prize.
    TEXT Congratulations!!! You've won a $1000 Walmart gift card!!! Click here to claim your prize.
    




    'Spam'


