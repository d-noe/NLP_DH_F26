"""
Data files to be downloaded from:
    https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/0TJX8Y
For this script, the files are to be stored in the folder:
    `data/data_dev/UNGDC/TXT/`

Within this folder, the files storage have the following structure (from download):

UNGDC_RAW_PATH/                # Main root folder containing all session folders
├── Session 01 - 1946/         # Example session folder
│   ├── USA_01_1946.txt        # Speech file for USA (Session 01, Year 1993)
│   ├── FRA_01_1946.txt        # Speech file for France
│   └── ...                    # Other country files
│
├── Session 02 - 1947/
│   ├── USA_02_1947.txt
│   ├── DEU_02_1947.txt
│   └── ...
│
├── Session 03 - 1948/
│   ├── UK_03_1948.txt
│   ├── BRA_03_1948.txt
│   └── ...
│
└── ...                        # Additional session folders


This script produces a two .csv files:
    - one that contain the speeches, ISO code of the speaker, year, session (note year <=> session)
    - one that breaks down the speeches into paragraphs based on coarse heuristics
"""
import os 
import pandas as pd
import re

# ==========================================
DATA_PATH = "../data_dev/UNGDC/TXT/"

OUTPUT_FLNMS = "../topic_data/ungdc.csv", "../topic_data/ungdc_coarse_paragraphs.csv"
# ==========================================

def parse_filename(
    filename:str,
):
    """
    Extract ISO code, session number, and year from filenames
    following the pattern <ISO>_<XX>_<YYYY>.txt
    
    Returns:
        dict: {
            "iso_code": str,
            "session": str,
            "year": str,
            "success": bool,
            "error": str or None
        }
    """
    pattern = r"^([A-Z]{2,3})_(\d{2})_(\d{4})\.txt$"
    match = re.match(pattern, filename)
    
    if match:
        iso_code, session, year = match.groups()
        return {
            "iso_code": iso_code,
            "session": session,
            "year": year,
            "success": True,
            "error": None
        }
    else:
        return {
            "iso_code": None,
            "session": None,
            "year": None,
            "success": False,
            "error": f"Filename '{filename}' does not match pattern '<ISO>_<XX>_<YYYY>.txt'"
        }

def get_ungdc_df(
    folders:list,
    base_path:str=DATA_PATH,
):
    # Initialize a list to collect rows
    records = []

    for curr_folder in folders:  # Loop over folders
        folder_path = os.path.join(base_path, curr_folder)
        for flnm in os.listdir(folder_path):  # Loop over files in each folder
            parsed_res = parse_filename(flnm)
            
            if parsed_res["success"]:
                curr_iso = parsed_res["iso_code"]
                curr_session = parsed_res["session"]
                curr_year = parsed_res["year"]
                
                file_path = os.path.join(folder_path, flnm)
                with open(file_path, 'r', encoding='utf-8') as f:
                    curr_text = f.read().strip()
                
                # Append one record (as a dict)
                records.append({
                    "session": curr_session,
                    "year": curr_year,
                    "country_iso": curr_iso,
                    "text": curr_text
                })
            else:
                print(f"⚠️ Skipping file: {flnm} — {parsed_res['error']}")

    # Build the DataFrame once, outside the loop
    df = pd.DataFrame(records, columns=["session", "year", "country_iso", "text"]).sort_values(by="year")
    return df

def remove_leading_numbers(
    text:str,
)->str:
    # Remove indices at the beginning of lines.
    text = re.sub(r'(?m)^\s*\d+\.\s*', '', text)

    return text

def clean_line_end_whitespace(text: str) -> str:
    """
    Remove trailing whitespace characters at the end of lines
    that follow sentence-ending punctuation, to avoid accidental merging.
    
    Example:
        "This is a line.   \nNext line." -> "This is a line.\nNext line."
    """
    # Remove spaces or tabs after punctuation at the end of a line
    text = re.sub(r'([.?!])\s+\n', r'\1\n', text)
    return text

def join_broken_lines(
    text:str
)->str:
    # Replace newlines that do NOT follow sentence-ending punctuation with a space
    text = re.sub(r'(?<![.?!])\n\s*', ' ', text)
    # Normalize spacing
    text = re.sub(r'([^\S\r\n]+)', ' ', text)  # replace multiple spaces/tabs with single space
    return text.strip()

def preprocess_txt(
    text:str
)->str:
    return remove_leading_numbers(
        join_broken_lines(
            clean_line_end_whitespace(
                text
            )
        )
    )

def split_speeches_to_paragraphs(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """
    Split speeches into paragraphs based on line breaks.
    Each paragraph gets a paragraph_id, and all other columns are repeated.
    
    Args:
        df: DataFrame containing speeches.
        text_col: Name of the column containing full speech text.
    
    Returns:
        DataFrame with columns: original columns + paragraph_id + text (paragraph)
    """
    records = []

    for _, row in df.iterrows():
        speech_text = row[text_col]
        paragraphs = [p.strip() for p in speech_text.split('\n') if p.strip()]
        for i, para in enumerate(paragraphs, start=1):
            new_row = row.to_dict()
            new_row[text_col] = para
            new_row["paragraph_id"] = i
            records.append(new_row)
    
    df_paragraphs = pd.DataFrame(records)
    return df_paragraphs


if __name__ == '__main__':
    print("Preprocessing UNGDC...")

    # Select Folders that correspond to the predefined pattern
    pattern = r"^Session\s\d{2}\s-\s\d{4}$"
    session_folders = [f for f in os.listdir(DATA_PATH) if re.match(pattern, f)]
    print(f"Total Number of folders: {len(session_folders)}\nExample of folders retrieved: {', '.join(session_folders[:3])}")

    df = get_ungdc_df(folders=session_folders)
    df.to_csv(OUTPUT_FLNMS[0], index=None)
    print(f"Pre-processed file [full speeches] (with {len(df)} speeches) saved at: {OUTPUT_FLNMS[0]}.")
    print("Splitting into paragraphs....")

    df_processed = df.copy()
    df_processed["text"] = df_processed["text"].apply(preprocess_txt)

    df_paragraphs = split_speeches_to_paragraphs(df_processed)

    df_paragraphs.to_csv(OUTPUT_FLNMS[1], index=None)
    print(f"Pre-processed file [paragraph-splitted] (with {len(df_paragraphs)} paragraphs) saved at: {OUTPUT_FLNMS[1]}.")