# Introduction to Natural Language Processing (NLP) — DH PSL,  Fall 2026

This repository hosts material for 6x3hours lectures in the context of the *Introduction to Natural Language Processing (NLP)* class from PSL's Master of Digital Humanities, Fall 2026.

1. [Week 1 (30/09)](#week1)
2. [Week 2 (07/10)](#week2)
3. [Week 3 (14/10)](#week3)
4. [Week 4 (21/10)](#week4)
5. [Week 5 (04/11)](#week5)
6. [Week 6 (18/11)](#week6)

The code and notebooks for the tutorials and hands-on sessions are provided in the [code](./code/) folder. The data used for these sessions is described and stored in [data](./data/).

## Week 1 (30/09): *Introduction* <a name="week1"></a>

- Slides: [preview `html`](https://rawcdn.githack.com/d-noe/NLP_DH_F26/refs/heads/main/slides/1_lecture_30_09_self_contained.html), [`pdf`](./slides/1_lecture_30_09.pdf) (until Tokenization Motivation, ~ slide 67)
- Notebook(s): [Annotation Workshop](./code/1_introduction/Tutorial_1_Annotation.ipynb)
- Key notions: linguistics basics & ambiguities, NLP historical eras, data-driven science, corpus, annotation


## Week 2 (07/10): *Word Representation* <a name="week2"></a>

- Slides: [preview `html`](https://rawcdn.githack.com/d-noe/NLP_DH_F26/refs/heads/main/slides/2_lecture_07_10_self_contained.html) [`pdf`](./slides/2_lecture_07_10.pdf)
- Notebook(s): [Tokenization](./code/2_word_representation/Tutorial_2_tokenization.ipynb); [Semantic Shift Analysis with word2vec](./code/2_word_representation/Hands_on_2_word2vec_semantic_shift.ipynb)
- Key notions: tokens, distributional semantics, semantic spaces, word2vec, cosine similarity

<details><summary>To go further</summary>

- [(Lenci, 2018)](https://pdfs.semanticscholar.org/3b6a/0fe7d4a254d2864f9643e80aeea188a28e81.pdf): *Distributional Models of Word Meaning.* — A review of different types of semantic models, mainly from a linguist perspective.
- [(Rong, 2016)](https://arxiv.org/pdf/1411.2738): *word2vec Parameter Learning Explained*

**Word2Vec & Biases**

- [(Caliskan, Bryson & Narayanan, 2017)](https://arxiv.org/pdf/1608.07187): *Semantics derived automatically from language corpora contain human-like biases* (introduce the famous WEAT (Word Embedding Association Test) evaluation)
- [(Garg et al., 2018)](https://doi.org/10.1073/pnas.1720347115): *Word Embeddings Quantify 100 Years of Gender and Ethnic Stereotypes.*
- [(Bolukbasi et al., 2016)](https://proceedings.neurips.cc/paper_files/paper/2016/file/a486cd07e4ac3d270571622f4f316ec5-Paper.pdf): *Man Is to Computer Programmer as Woman Is to Homemaker? Debiasing Word Embeddings.* (also introduce a "debiasing" mechanism)

**Word2Vec & Semantic Shifts**

- [(Hamilton, Leskovec & Jurafsky, 2016)](https://aclanthology.org/P16-1141.pdf): *Diachronic word embeddings reveal statistical laws of semantic change*

</details>


## Week 3 (14/10): *Language Modeling* <a name="week3"></a>

- Slides: [preview `html`](https://rawcdn.githack.com/d-noe/NLP_DH_F26/refs/heads/main/slides/3_lecture_14_10_self_contained.html) [`pdf`](./slides/3_lecture_14_10.pdf)
- Notebook(s): [BERT Discovery](./code/3_bert_training/Tutorial_3a_Discover_BERT.ipynb), [Word Sense Disambiguation](./code/3_bert_training/Tutorial_3b_WSD.ipynb), [Semantic Shifts](./code/3_bert_training/Hands_on_3_SS.ipynb), [[bonus] NER](./code/3_bert_training/Guided_lab_3_NER.ipynb)
- Key notions: n-gram, transformers, self-attention, contextual representations, masked language model / causal language model, BERT

<details><summary>To go further</summary>

- [(J. Alammar, 2018)](https://jalammar.github.io/illustrated-transformer/): *The Illustrated Transformer* blog post by Jay Alammar.
- [(Ghaseminejad Raeini, 2025)](https://www.sciencedirect.com/science/article/pii/S2949719125000445): *The evolution of language models: From N-Grams to LLMs, and beyond*.


**Named Entity Recognition**

- [(Ehrmann et al., 2023)](https://dl.acm.org/doi/full/10.1145/3604931): *Named Entity Recognition and Classification in Historical Documents: A Survey*

**Want more hands-on?** Check the [*To go further* section in code folder](./code/README.md#code_supp_1).

</details>

## Week 4 (21/10): *Discovering structure* <a name="week4"></a>

- Slides: TBA
- Notebook(s): TBA
- Key notions: TBA

## Week 5 (04/11): *Learning Patterns* <a name="week5"></a>

- Slides: TBA
- Notebook(s): TBA
- Key notions: TBA

## Week 6 (18/11): *LLMs* <a name="week6"></a>

- Slides: TBA
- Notebook(s): TBA
- Key notions: TBA