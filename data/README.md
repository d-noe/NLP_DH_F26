# Data

This folder hosts the data used in the different tutorials and hands-on sessions. Short descriptions are provided below, please refer to the original sources for further information.

- [Week 1](#data_week1)

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
