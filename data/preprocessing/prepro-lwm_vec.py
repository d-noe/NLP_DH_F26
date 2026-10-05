"""
Data files to be downloaded from:
    https://zenodo.org/records/7887305
For this script, the following files:
    1800s.model
	1800s.model.syn1neg.npy
	1800s.model.wv.vectors.npy
	1820s.model
	1820s.model.syn1neg.npy
	1820s.model.wv.vectors.npy
	1840s.model
	1840s.model.syn1neg.npy
	1840s.model.wv.vectors.npy
	1860s.model
	1860s.model.syn1neg.npy
	1860s.model.wv.vectors.npy
	1880s.model
	1880s.model.syn1neg.npy
	1880s.model.wv.vectors.npy
	1900s.model
	1900s.model.syn1neg.npy
	1900s.model.wv.vectors.npy
need to be stored in the folder:
    `data/data_dev/lmw_wv_unaligned/`

For filtering, the file 'wiki-100k.txt' 
should also be downloded from https://gist.github.com/h3xx/1976236 
and stored at `data_dev/`

This script produces 6 .bin files: 
the pruned representations of the original historical word2vec embeddings
filtered using heuristics based on word frequency (in the corpus and common English words), and spell checks
"""

from gensim.models import KeyedVectors
from gensim.models import Word2Vec

from scipy import spatial
import pandas as pd
import numpy as np
import os

## For filtering
from spellchecker import SpellChecker
spell = SpellChecker()

with open("../data_dev/wiki-100k.txt") as f:
    wiki_100k = f.readlines()
f.close()
wiki_100k = [w[:-1].lower() for w in wiki_100k if (not w.startswith('#'))]

top_k_per_decade = 50000  # Extract top 50k words by file rank
min_decade_presence = .8 # Word must appear in at least 80% of decades

do_top_n = True # filter by most frequent words
do_100k_filter = 30000 # filter based on 30k most frequent English words (modern! based on wikipedia)
do_spell_check = True # run spell check (same issue?)

##

BASE_PATH = "../data_dev/lwm_wv_unaligned"
OUTPUT_FOLDER = "../lwm_wordvecs/"


def get_filtered_kv(model, voc):
    words = [
        word for word in model.wv.index_to_key
        if word in voc
    ]
    # Create a new KeyedVectors object
    reduced_kv = KeyedVectors(
        vector_size=model.wv.vector_size
    )
    # Add only the selected vectors
    reduced_kv.add_vectors(
        words,
        model.wv.vectors[
            [model.wv.key_to_index[word] for word in words]
        ]
    )
    return reduced_kv

if __name__ == "__main__":
    print("Filtering the vocabulary of Living With the Machines' pretrained word2vec diachronic representations")

    years = sorted([f.split(".")[0] for f in os.listdir(BASE_PATH) if f.endswith("model")])
    print(f"The folder contains models for the decades: {years}.\nLoading the models...")
    loaded_models = {
        y: Word2Vec.load(os.path.join(BASE_PATH, f"{y}.model"))
        for y in years
    }
    print(f"... models loaded!")

    decade_top_vocabs = {}

    print("Step 1: Inspecting vocabulary order...")
    for y in years:
        if do_top_n:
            top_keys = loaded_models[y].wv.index_to_key[:top_k_per_decade]
        else:
            top_keys = loaded_models[y].wv.index_to_key
        decade_top_vocabs[y] = set(top_keys)
        print(f"  {y}: extracted top {len(top_keys):,} words")

    # 2. Build the shared vocabulary across decades
    all_words = [w for vocab in decade_top_vocabs.values() for w in vocab]
    word_counts = {}
    for w in all_words:
        word_counts[w] = word_counts.get(w, 0) + 1

    threshold_count = int(len(years) * min_decade_presence)
    shared_vocab = {w for w, count in word_counts.items() if count >= threshold_count}

    print(f"\nStep 2: Shared vocabulary built. Total words: {len(shared_vocab):,}")

    if do_spell_check:
        known = set(spell.word_frequency.keys())
        shared_vocab = known.intersection(shared_vocab)
        print(f"- spell check done, vocabulary reduced to: {len(shared_vocab):,}")
    else:
        print("No spell check.")

    if do_100k_filter > 0:
        f_words = set(wiki_100k[:do_100k_filter])
        shared_vocab = {w for w in shared_vocab if w in f_words}
        print(f"- filtering by {do_100k_filter} most frequent English words, vocabulary to {len(shared_vocab):,}")
    else:
        print("No filtering by most common English word")

    # 3. Prune vector representation
    print(f"Filtering the vector representations...")
    reduced_kvs = {
        k: get_filtered_kv(m, voc=shared_vocab)
        for k, m in loaded_models.items()
    }

    final_vocab = shared_vocab

    print("Filtering done.\nStep 3: Exporting pruned vector files...")
    for y, kv in reduced_kvs.items():
        pruned_kv = KeyedVectors(vector_size=kv.vector_size)

        curr_words = list(kv.key_to_index.keys())
        vectors = [kv[w] for w in curr_words]
        pruned_kv.add_vectors(curr_words, vectors)
        
        out_filename = f"pruned_{y}.bin"
        out_path = os.path.join(OUTPUT_FOLDER, out_filename)
        pruned_kv.save_word2vec_format(out_path, binary=True)
        
        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        print(f"  Saved {out_filename} ({len(kv.key_to_index.keys())} words, {size_mb:.1f} MB)")

    print("Filtering done.")