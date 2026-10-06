# Introduction to Natural Language Processing (NLP) — DH PSL,  Fall 2026

This repository hosts material for 6x3hours lectures in the context of the *Introduction to Natural Language Processing (NLP)* class from PSL's Master of Digital Humanities, Fall 2026.

1. [Week 1 (30/09)](#week1)

The code and notebooks for the tutorials and hands-on sessions are provided in the [code](./code/) folder. The data used for these sessions is described and stored in [data](./data/).

## Week 1 (30/09): *Introduction* <a name="week1"></a>

- Slides: [preview `html`](https://rawcdn.githack.com/d-noe/NLP_DH_F26/refs/heads/main/slides/lecture_30_09_self_contained.html), [`pdf`](./slides/lecture_30_09.pdf) (until Tokenization Motivation, ~ slide 67)
- Notebook(s): [Annotation Workshop](./code/1_introduction/Tutorial_1_Annotation.ipynb)
- Key notions: linguistics basics & ambiguities, NLP historical eras, data-driven science, corpus, annotation

<details><summary>To go further</summary>

</details>


## Week 2 (07/10): *Introduction* <a name="week2"></a>

- Slides: soon
- Notebook(s): [Tokenization](./code/2_word_representation/Tutorial_2_tokenization.ipynb); [Semantic Shift Analysis with word2vec](./code/2_word_representation/Hands_on_2_word2vec_semantic_shift.ipynb)
- Key notions: tokens, distributional semantics, semantic spaces, word2vec, cosine similarity

<details><summary>To go further</summary>

- [(Lenci, 2018)](https://pdfs.semanticscholar.org/3b6a/0fe7d4a254d2864f9643e80aeea188a28e81.pdf): *Distributional Models of Word Meaning.* — A review of different types of semantic models, mainly from a linguist perspective.

**Word2Vec & Biases**

- [(Caliskan, Bryson & Narayanan, 2017)](https://arxiv.org/pdf/1608.07187): *Semantics derived automatically from language corpora contain human-like biases* (introduce the famous WEAT (Word Embedding Association Test) evaluation)
- [(Garg et al., 2018)](https://doi.org/10.1073/pnas.1720347115): *Word Embeddings Quantify 100 Years of Gender and Ethnic Stereotypes.*
- [(Bolukbasi et al., 2016)](https://proceedings.neurips.cc/paper_files/paper/2016/file/a486cd07e4ac3d270571622f4f316ec5-Paper.pdf): *Man Is to Computer Programmer as Woman Is to Homemaker? Debiasing Word Embeddings.* (also introduce a "debiasing" mechanism)

**Word2Vec & Semantic Shifts**

- [(Hamilton, Leskovec & Jurafsky, 2016)](https://aclanthology.org/P16-1141.pdf): *Diachronic word embeddings reveal statistical laws of semantic change*

</details>
