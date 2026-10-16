"""
Data files to be downloaded from:
    https://github.com/ViralTexts/nineteenth-century-recipes/tree/main/plaintext_recipes
For this script, the files are to be stored in the folder:
    `data/data_dev/ViralTexts-nineteenth-century-recipes-plaintext/`

This script produces a .csv file that stores the texts (and original src: Project Gutenberg id)
"""
import os 
import pandas as pd

DATA_PATH = "../data_dev/ViralTexts-nineteenth-century-recipes-plaintext/"

OUTPUT_FLNM = "../topic_data/nineteenth_recipes.csv"


def get_flnms(
    base_path:str=DATA_PATH,
)->list:
    flnms = []
    for f in os.listdir(base_path):
        if f.endswith('.txt'):
            flnms += [f]
    return flnms

def read_files(
    filenames:list,
    base_path:str=DATA_PATH,
)->list:
    contents = []
    for f in filenames:
        with open(os.path.join(base_path, f), "r") as recipe:
            contents += [recipe.read()]
        recipe.close()
    return contents

def get_gutenberg_ids(
    filenames:list,
):
    return [f.split("_")[0] for f in filenames]


if __name__ == '__main__':
    print("Preprocessing Nineteenth Century Recipes")

    flnms = get_flnms()
    texts = read_files(flnms)
    ids = get_gutenberg_ids(flnms)

    df = pd.DataFrame()
    df["text"] = texts
    df["ids"] = ids

    df.to_csv(OUTPUT_FLNM, index=None)
    
    print(f"Pre-processed file (with {len(df)} recipes) saved at: {OUTPUT_FLNM}.")
