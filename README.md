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
Explore o Jupyter Notebook em [SpamDetectionBERT](SpamDetetcionBERT.ipynb)
