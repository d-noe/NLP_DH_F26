# Code

This folder hosts the notebooks and code (in Python) used in the different tutorials and hands-on sessions. The proposed [set-ups](#setup), and [contents](#content) of the sessions are described below.


- [Week 1](#code_week1): Introduction — 1. familiarize yourself with the annotation process, and 2. explore text tokenization through off-the-shelf libraries or a custom implementation of the BPE algorithm.

## Setups <a name="setup"></a>

Feel free to use the notebooks, either locally, or using hosted services such as Jupyter Binder and Google Colab.

### Running on your machine

You can use the `requirements.txt` file provided at the root of this repository. In your virtual environment, `cd` to repo root, and run:

```bash
pip install -r requirements.txt
```

### Binder

You can launch the projects on Binder: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/d-noe/NLP_DH_F26/HEAD)

> [!WARNING]
> It can take some time to build the image on Binder.

Binder can be handy to have the repository in a jupyter-lab hosted environment. However, it does not provide extensive memory nor computational resources. Thus, it is not fitted to manipulate large data or to use pre-trained language models with large number of parameters.

### Colab

The notebooks are provided in Google Colab. It provides a convenient way to run the experiments and offers computational resources that should be sufficient for the content of this course in the free-tier (including GPU and TPU runtime access).



## Content <a name="content"></a>

### Week 1 — 30.09 <a name="code_week1"></a>

- [Tutorial_1_Annotation.ipynb](./1_introduction/Tutorial_1_Annotation.ipynb): Workshop on data annotation and interannotator agreement exploration.
~~- [Hands_on_1_tokenization.ipynb](./1_introduction/Hands_on_1_tokenization.ipynb): Explore tokenization using different methods. Implement your own version of the BPE algorithm, and re-discover Zipf's law.~~ (next week)

**Main libraries**: `nltk`, `spacy`, `tiktoken` (`pigeonxt-jupyter`, `pandas`, `statsmodels`, `krippendorff`)

<a name="code_supp_1"></a>
<details><summary>To go further</summary>


</details>
