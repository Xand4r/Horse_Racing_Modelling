import csv
import os
from bs4 import BeautifulSoup

# Paths are relative to the repository root, so run this script from there
# (e.g. `python src/data/parser.py`).
DATA_DIR = "src/data"
RAW_DIR = f"{DATA_DIR}/raw_racecards"
PARSED_DIR = f"{DATA_DIR}/parsed_data"


def extract_classno_distance(info: list[str]) -> tuple[str, str]:
    class_no = ""
    distance = ""
    for i in range(len(info)):
        if info[i] == "-":
            distance = info[i + 1]
            break
        class_no += info[i] + " "
    class_no = class_no.strip()
    return class_no, distance

def parse_race_info(directory: str) -> None:

    with open(f"{PARSED_DIR}/races.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["id", "date", "venue", "class", "distance", "going", "course"])

        for race in os.scandir(directory):
            if not race.name.endswith(".html"):
                continue
            with open(race.path, "r", encoding="utf-8") as file:
                soup = BeautifulSoup(file, "html.parser")

            race_meeting = soup.find("span", "f_fl f_fs13").get_text(strip=True)
            date = race_meeting.split()[2]
            venue = "".join(race_meeting.split()[-2:])

            race_info = soup.find("tbody", "f_fs13")
            details = race_info.find_all("tr")
            inforow_1 = details[1].get_text(strip=True).split()
            inforow_2 = details[2].get_text(strip=True).split()

            class_no, distance = extract_classno_distance(inforow_1)

            going = "".join(inforow_1).partition(":")[2]

            course = "".join(inforow_2).partition(":")[2].replace('"', "")


            race_id = os.path.splitext(race.name)[0]
            #add padding to race number in id
            if len(race_id) == 17:
                prefix, no = race_id.rsplit("_", 1)
                race_id = prefix + "_" + no.zfill(2)

            writer.writerow([race_id, date, venue, class_no, distance, going, course])

    return None

def parse_result_info(directory : str) -> None:
    with open(f"{PARSED_DIR}/results.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["id", "place", "horse_no", "name", "jockey", "trainer", "carried_weight", "horse_weight",
                         "draw", "LBW", "section_1", "section_2", "section_3", "section_4", "section_5", "section_6",
                         "finish_time", "odds"])

        for race in os.scandir(directory):
            if not race.name.endswith(".html"):
                continue

            with open(race.path, "r", encoding="utf-8") as file:
                soup = BeautifulSoup(file, "html.parser")

            result_table = soup.find("tbody", "f_fs12")
            for row in result_table.find_all("tr"):

                race_id = os.path.splitext(race.name)[0]
                # add padding to race number in id if only one digit
                if len(race_id) == 17:
                    prefix, no = race_id.rsplit("_", 1)
                    race_id = prefix + "_" + no.zfill(2)

                data = [race_id]
                for td in row.find_all("td"):
                    parent_divs = td.find("div")
                    if parent_divs:
                        for div in parent_divs.find_all("div"):
                            data.append(div.get_text(strip=True))
                        for i in range(6 - len(parent_divs.find_all("div"))):
                            data.append("-")
                        continue
                    data.append(td.get_text(strip=True))
                writer.writerow(data)
    return None


if __name__ == "__main__":
    # Run from the repository root. Reads the scraped HTML in src/data/raw_racecards/
    # and writes the flat tables src/data/parsed_data/races.csv and results.csv.
    parse_race_info(RAW_DIR)
    parse_result_info(RAW_DIR)



