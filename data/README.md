# Data

This folder hosts the data used in the different tutorials and hands-on sessions. Short descriptions are provided below, please refer to the original sources for further information.

- [Week 1](#data_week1)
- [Week 2](#data_week2)
    - Semantic Shifts Analysis: [Living With Machines - Vectors](#lwm_vectors)
- [Week 3](#data_week3)
  - Word Sense Disambiguation: [CoarseWSD-20](#wsd)
  - Semantic Shifts Analysis: [Living With Machines - Texts](#lwm)
  - BERT-based Named Entity Recognition (NER): [CLEF-HIPE-2020](#hipe)

The [`preprocessing`](./preprocessing/) folder contains pre-processing scripts used to obtain the presented data files. It is included for transparency and to allow reproduction, but was not developed for pedagogical purposes (-> the code is likely quick and dirty).

## Week 1 — 30.09 <a name="data_week1"></a>

### Tutorial 1: Sentiment Annotation Workshop

- File(s): [`annotation_data/sentences_to_annot.csv`](./annotation_data/sentences_to_annot.csv), [`annotation_data/sentences_with_annot.csv`](./annotation_data/sentences_with_annot.csv), 
- Description:
    - A small scale (n=30) sample of short sentences (based on 'AmbiSent dataset' (version 'AmbiSent_Minimal_n53.csv') and sampled through a label-entropy heuristic) for a sentiment annotation task. The different versions of the files propose a version truncated from the original study's annotation outputs, one containing the original AmbiSent annotation and additional metadata; finally the annotation made by participants of the in-class workshop (see [here](../code/1_introduction/Tutorial_1_Annotation.ipynb)) are provided in the subfolder [`your_annotations`](./annotation_data/your_annotations/).
    - Columns:
        - `sentences_to_annot.csv`: `Sentence_id`, `Sentence`
        - `sentences_with_annot.csv`: `Sentence_id`, `Sentence`, `Positive`, `Negative`, `Mixed`, `Neutral`, `Correct`, `Sentence_type`
        - `your_annotations/<FLNM>.csv`: `Sentence_id`, `Sentence`, `Label`, `Positive`, `Negative`, `Mixed`, `Neutral`
- Source: Äyräväinen, L. E., Hinds, J., & Davidson, B. I. (2025). Disambiguating sentiment annotation: A mixed methods investigation of annotator experience and impact of instructions on annotator agreement. Plos one, 20(12), e0336269. [Paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0336269) | [Data Repository](https://osf.io/x687m/files/vjkdx)


## Week 2 — 07.10 <a name="data_week2"></a>

### Hands-on 2: Semantic Shifts in 19th century newspapers  <a name="lwm_vectors"></a>

Data from the [Living With Machines](https://livingwithmachines.ac.uk) programme. Please check their website if interested in *digital history*, and to discover their research on rethinking the impact of technology on people's lives during the Industrial Revolution.

- File(s): Six binary files stored at [`lwm_wordvecs/`](./lwm_wordvecs/) - containing pretrained diachronic word2vec representations. 
- Description:
  - Each file, formated as `pruned_<DECADE>.bin`, stores the word2vec pretrained embedding computed on a time slice (indicated by the `<DECADE>`: 1800s, 1820s, 1840s, 1860s, 1880s, and 1900s) of a massive dataset of historical newspapers (roughly 4.6B tokens in total). The original representations are further pruned to reduce the size of the files (according to the [linked preprocessing script](preprocessing/prepro-lwm_vec.py)).
  - Each file stores `gensim.keyedvectors`: words from the vocabulary associated with a $d=200$-dimensional semantic vector.
- Source: Pedrazzini, N., & McGillivray, B. (2022). Machines in the media: semantic change in the lexicon of mechanization in 19th-century British newspapers. In Proceedings of the 2nd International Workshop on Natural Language Processing for Digital Humanities (pp. 85-95). [Link](https://aclanthology.org/2022.nlp4dh-1.12.pdf) ; [Code Repository](https://github.com/Living-with-machines/DiachronicEmb-BigHistData); [Data Repository](https://zenodo.org/records/7887305)

## Week 3 — 14.10 <a name="data_week3"></a>

### Tutorial 3b: Word Sense Disambiguation <a name="wsd"></a>

- File(s): [`word_sense/CoarseWSD-20.csv`](./word_sense/CoarseWSD-20.csv)
- Description:
  - Dataset based on Wikipedia articles excerpts, developed to evaluate and train word disambiguation models based on a selection of 20 expert-selected words (apple, club, bow, bank, crane, square, chair, java, bass, seal, pitcher, arm, mole, pound, deck, trunk, spring, hood, yard, digit).
  - columns: `text`, `target`, `label`, `label_sense`, `split`
- Source: *Daniel Loureiro, Kiamehr Rezaee, Mohammad Taher Pilehvar, & Jose Camacho-Collados. (2021). Analysis and Evaluation of Language Models for Word Sense Disambiguation.* [![arXiv](https://img.shields.io/badge/arXiv-2008.11608-b31b1b.svg)](https://arxiv.org/abs/2008.11608)

### Hands-on 3: Semantic Shifts <a name="lwm"></a>

Data from the [Living With Machines](https://livingwithmachines.ac.uk) programme. Please check their website if interested in *digital history*, and to discover their research on rethinking the impact of technology on people's lives during the Industrial Revolution.

- File(s): [`word_sense/lwm_annot.csv`](./word_sense/lwm_annot.csv)
- Description:
  - Excerpts of british historical newspaper articles (1818-1920), OCR texts, annotated for word sense for four target words (trolley, car, bike, coach).
  - columns: `text`, `annot`, `newspaper_date`, `newspaper_title`, `newspaper_place`, `image_url`, `image_url_conflict_values`, `target`, `year`.
- Source: Ridge, M., Pedrazzini, N., Vieira, J. M. M., Ciula, A., & McGillivray, B. (2024). Language of mechanisation crowdsourcing datasets from the living with machines project. Journal of Open Humanities Data, 10, 33. [Link](https://kclpure.kcl.ac.uk/portal/en/publications/language-of-mechanisation-crowdsourcing-datasets-from-the-living-) ; [Data Repository](https://bl.iro.bl.uk/concern/datasets/1a677294-cbd3-4bc0-b714-d3bbfd2a6da1)


### Guided Lab 3 [bonus]: BERT-based Named Entity Recognition (NER) <a name="hipe"></a>

- File(s): [`hipe_ner/prepro_hipe_fr/prepro_hipe_fr/`](./hipe_ner/prepro_hipe_fr/) (Hugging Face `DatasetDict` pyarrow format)
- Description:
  - OCR-processed French historical primary sources (19th and 20th-century Swiss, French, and Luxembourgish newspapers) compiled for the CLEF-HIPE-2020 shared task.
  - An arrow-formatted Hugging Face `DatasetDict` containing preprocessed French data from the CLEF-HIPE-2020 shared task (version 1.4). The data was segmented using PySDB annotations and filtered by length to retain only sequences between 8 and 128 tokens. It is structured into `train`, `dev`, and `test` splits optimized for named entity recognition (NER) tasks, with unified label vocabularies assigned consistently across all splits.
  - Columns / Features:
    - `train`, `dev`, `test` splits: `tokens`, `ner`, `meto`
- Source: Ehrmann, M., Romanello, M., Bircher, S., Clematide, S. (2020). Introducing the CLEF 2020 HIPE Shared Task: Named Entity Recognition and Linking on Historical Newspapers. In: Jose, J., et al. Advances in Information Retrieval. ECIR 2020. Lecture Notes in Computer Science(), vol 12036. Springer, Cham. [Link](https://doi.org/10.1007/978-3-030-45442-5_68) | [Repository](https://www.google.com/search?q=https://github.com/impresso/CLEF-HIPE-2020)
