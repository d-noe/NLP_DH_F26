# Data

This folder hosts the data used in the different tutorials and hands-on sessions. Short descriptions are provided below, please refer to the original sources for further information.

- [Week 1](#data_week1)
- [Week 2](#data_week2)
    - Semantic Shifts Analysis: [Living With Machines - Vectors](#lwm_vectors)

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

