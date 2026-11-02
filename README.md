# Introduction to Natural Language Processing (NLP) — DH PSL,  Fall 2026

This repository hosts material for 6x3hours lectures in the context of the *Introduction to Natural Language Processing (NLP)* class from PSL's Master of Digital Humanities, Fall 2026.

1. [Week 1 (30/09)](#week1)
2. [Week 2 (07/10)](#week2)
3. [Week 3 (14/10)](#week3)
4. [Week 4 (21/10)](#week4)
5. [Week 5 (04/11)](#week5)
6. [Week 6 (18/11)](#week6)

Slides are available in the [slides](./slides) folder, both in `pdf` and `html` formats. Please note that the `pdf` versions might be, at times, partially obstructed because of the animations in the original `html` format.

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

- Slides: [preview `html`](https://rawcdn.githack.com/d-noe/NLP_DH_F26/refs/heads/main/slides/4_lecture_21_10_self_contained.html) [`pdf`](./slides/4_lecture_21_10.pdf)
- Notebook(s): [Custom BERTopic](./code/4_topic_modeling/Tutorial_4_MyBERTopic.ipynb), [Topic Modeling UN General Debates Speeches](./code/4_topic_modeling/Hands_on_4_CompareTM.ipynb)
- Key notions: document representation, BoW, SentenceTransformer, topic modeling, BERTopic, LDA, PMI

<details><summary>To go further</summary>

**Dimensionality Reduction**:
- [(Coenen & Pierce, 2019)](https://pair-code.github.io/understanding-umap/): *Understanding UMAP*: explanations and visual demonstration of UMAP (compared with t-SNE).

**Topic Modeling**:
- [(Churchill & Singh, 2021)](https://doi.org/10.1145/3507900): *The Evolution of Topic Modeling*.
- [(Li et al., 2024)](https://doi.org/10.1515/dsll-2024-0010): *Applying Topic Modeling to Literary Analysis: A Review*.
- [(Gillings & Hardie, 2022)](https://doi.org/10.1093/llc/fqac075): *The interpretation of topic models for scholarly analysis: An evaluation and critique of current practice*.
- [(Antoniak, 2023)](https://maria-antoniak.github.io/2022/07/27/topic-modeling-for-the-people.html): *Topic Modeling for the People*: an interesting blogpost by Maria Antoniak, sharing a set of steps that you can follow to get coherent topics from most datasets, primarily focusing on LDA. It provides as well many additional references to dig deeper.
- [(Egger & Yu, 2022)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9120935/): *A Topic Modeling Comparison Between LDA, NMF, Top2Vec, and BERTopic to Demystify Twitter Posts*.
- **Evaluation Concerns**:
    - [(Chang et al., 2009)](https://papers.nips.cc/paper_files/paper/2009/hash/f92586a25bb3145facd64ab20fd554ff-Abstract.html): *Reading Tea Leaves: How Humans Interpret Topic Models*.
    - [(Hoyle et al., 2021)](https://proceedings.neurips.cc/paper/2021/file/0f83556a305d789b1d71815e8ea4f4b0-Paper.pdf): *Is Automated Topic Model Evaluation Broken?: The Incoherence of Coherence*.

**Want more hands-on?** Check the [*To go further* section in code folder](./code/README.md#code_supp_2).

</details>

## Week 5 (04/11): *Learning Patterns* <a name="week5"></a>

- Slides: TBA
- Notebook(s): [BERT Fine-Tuning Tutorial](./code/5_supervised/Tutorial_5_SFT.ipynb), [Canonicity Prediction Challenge: Performance and Fairness](./code/5_supervised/Hands_on_5_CanonChallenge.ipynb)
- Key notions: supervised fine-tuning, classification, performance metrics, fairness

<details><summary>To go further</summary>

**Text Classification for DH**
- [(Bamman et al., 2024)](https://ceur-ws.org/Vol-3834/paper119.pdf): *On Classification with Large Language Models in Cultural Analytics*.
- [(Lassen et al., 2024)](https://ceur-ws.org/Vol-3834/paper76.pdf): *Literary Canonicity and Algorithmic Fairness: The Effect of Author Gender on Classification Models*.

**Fairness & Bias**
- [(Solon Barocas, Moritz Hardt, Arvind Narayanan, 2023)](https://fairmlbook.org): *FAIRNESS AND MACHINE LEARNING - Limitations and Opportunities* — Full book available with additional resources.
- [(Irving & Askell, 2019)](10.23915/distill.00014): *AI Safety Needs Social Scientists*.
- [(Blodgett et al., 2020)](https://aclanthology.org/2020.acl-main.485/): *Language (Technology) is Power: A Critical Survey of ``Bias'' in NLP.*
- [(Hovy & Prabhumoye)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9285808/pdf/LNC3-15-0.pdf): *Five sources of bias in natural language processing*.
- [(Gallegos et al., 2024)](https://aclanthology.org/2024.cl-3.8.pdf): *Bias and Fairness in Large Language Models: A Survey*.

**Interpretability**
- [Interpretability Blog-post](https://blog.ml.cmu.edu/2020/08/31/6-interpretability/)
- [(Olah et al., 2018)](10.23915/distill.00010): *The Building Blocks of Interpretability*: Mainly focusing (or applying) on computer vision, but a very nice and illustrated article on interpretability. 

**Want more hands-on?** Check the [*To go further* section in code folder](./code/README.md#code_supp_5).

</details>

## Week 6 (18/11): *LLMs* <a name="week6"></a>

- Slides: TBA
- Notebook(s): TBA
- Key notions: TBA