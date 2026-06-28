import numpy as np
import pandas as pd

# Paths are relative to the repository root, so run this script from there
# (e.g. `python src/data/cleaning.py`).
DATA_DIR = "src/data"
PARSED_DIR = f"{DATA_DIR}/parsed_data"
CLEANED_DIR = f"{DATA_DIR}/cleaned_data"

def distance_formatter(distance: str) -> int:
    res = distance[:-1]
    return int(res)

def lbw_converter(lbw: str) -> str:
    if lbw == "-" or lbw == "---":
        return "0"
    elif lbw == "NOSE":
        return "0.11"
    elif lbw == "SH":
        return "0.22"
    elif lbw == "HD":
        return "0.44"
    elif lbw == "N":
        return "0.66"
    elif lbw == "ML":
        return "440"

    if "-" in lbw:
        product = lbw.partition("-")
        fraction = product[2]
        numerator = fraction[0]
        denominator = fraction[2]
        meters = float(product[0]) * 2.2 + float(numerator) / float(denominator) * 2.2
        return str(round(meters,2))

    if "/" in lbw:
        fraction = lbw.partition("/")
        numerator = fraction[0]
        denominator = fraction[2]
        meters = float(numerator) / float(denominator) * 2.2
        return str(round(meters,2))

    return lbw


def clean_races() -> None:
    df = pd.read_csv(f"{PARSED_DIR}/races.csv", header=0)
    df['class'] = df['class'].replace({"Class 1 (Restricted)": "Class 1", "Class 2 (Restricted)": "Class 2",
                         "Class 3 (Restricted)": "Class 3", "Class 4 (Restricted)": "Class 4",
                         "Class 5 (Restricted)": "Class 5"})

    df['class'] = df['class'].replace({"Class 1": 1, "Class 2": 2,
                                       "Class 3": 3, "Class 4": 4,
                                       "Class 5": 5})
    df['distance'] = df['distance'].apply(lambda x: distance_formatter(x))
    df.to_csv(f"{CLEANED_DIR}/cleaned_races.csv", index=False, header=True)



def clean_results() -> None:
    df = pd.read_csv(f"{PARSED_DIR}/results.csv", header=0)
    #summarize dead heats
    df['place'] = df['place'].replace({"1 DH": "1", "2 DH": "2", "3 DH": "3", "4 DH": "4", "5 DH": "5", "6 DH": "6",
                                       "7 DH": "7", "8 DH": "8", "9 DH": "9", "10 DH": "10", "11 DH": "11",
                                       "12 DH": "12", "13 DH": "13", "14 DH": "14", "15 DH": "15"})

    #remove non-placements
    df = df[df['place'].isin(["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"])]

    df['section_1'] = df['section_1'].replace("-", np.nan)
    df['section_2'] = df['section_2'].replace("-", np.nan)
    df['section_3'] = df['section_3'].replace("-", np.nan)
    df['section_4'] = df['section_4'].replace("-", np.nan)
    df['section_5'] = df['section_5'].replace("-", np.nan)
    df['section_6'] = df['section_6'].replace("-", np.nan)

    df['LBW'] = df['LBW'].apply(lambda x: lbw_converter(x))


    df.to_csv(f"{CLEANED_DIR}/cleaned_results.csv", index=False, header=True)

if __name__ == "__main__":
    # Run from the repository root after parser.py. Reads src/data/parsed_data/ and
    # writes the numeric, model-ready tables into src/data/cleaned_data/. The scraped
    # data usually needs some manual inspection/cleanup before this step (see README).
    clean_results()
    clean_races()