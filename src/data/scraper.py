import requests
from bs4 import BeautifulSoup
import csv
import time
import os
import sys

# Paths are relative to the repository root, so run this script from there
# (e.g. `python src/data/scraper.py`).
DATA_DIR = "src/data"
DATES_CSV = f"{DATA_DIR}/dates.csv"
RAW_DIR = f"{DATA_DIR}/raw_racecards"


def convert_date(date: str) -> str:
    res = ""
    temp = ""
    for char in date:
        if char == "/":
            res = "/" + temp + res
            temp = ""
            continue

        temp += char

    return temp + res


def scrape_dates() ->  None:
    res = []
    url = "https://racing.hkjc.com/en-us/local/information/localresults"
    response = requests.get(url)
    if response.status_code != 200:
        return None

    soup = BeautifulSoup(response.text, "html.parser")
    select = soup.find(id = "selectId")

    for option in select.find_all("option"):
        val = option.get_text(strip=True)
        if not val:
            continue
        val = convert_date(val)

        checkresponse = requests.get(f"https://racing.hkjc.com/en-us/local/information/localresults?racedate={val}",
                                     allow_redirects=False)
        if checkresponse.status_code == 200:
            res.append([val])
    with open(DATES_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(res)

    return None

def scrape_data() -> None:

    url = "https://racing.hkjc.com/en-us/local/information/localresults?racedate="
    dates = open(DATES_CSV, "r")
    reader = csv.reader(dates)
    for row in reader:
        if row[0] == "#":
            break
        response = requests.get(url+row[0])
        if response.status_code != 200:
            continue
        soup = BeautifulSoup(response.text, "html.parser")
        races = soup.select("table.js_racecard tr td")
        race_count = len(races) - 2
        for race in range(race_count):
            race_card = requests.get(url+row[0]+"&RaceNo="+str(race+1))
            if race_card.status_code != 200:
                continue
            time.sleep(1)
            safe_date = row[0].replace("/", "-")
            filepath = f"{RAW_DIR}/{safe_date}_race_{race+1}.html"
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(race_card.text)

    return None


if __name__ == "__main__":
    # Run from the repository root. Two separate steps (both hit the HKJC servers):
    #
    #   python src/data/scraper.py dates
    #       discover the available race dates -> src/data/dates.csv
    #
    # You can then edit src/data/dates.csv by hand: add specific race dates, or put a
    # "#" marker row to stop scraping at a chosen point (scrape_data stops when it
    # reaches the "#"). Then download the race cards with:
    #
    #   python src/data/scraper.py
    #       read dates.csv -> save raw HTML into src/data/raw_racecards/
    if (len(sys.argv) > 1 and sys.argv[1] == "dates") or not os.path.exists(DATES_CSV):
        scrape_dates()
    else:
        scrape_data()