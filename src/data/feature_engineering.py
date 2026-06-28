import numpy as np
import pandas as pd
from datetime import date

def horse_winrate(past_info: pd.DataFrame, moving_window_size: int) -> tuple[float, int] | tuple[None, None]:
    possible = min(moving_window_size, past_info.shape[0])
    if possible == 0:
        return None, None
    needed_timeframe = past_info.tail(possible)
    wins = needed_timeframe[needed_timeframe["place"] == 1].shape[0]
    return wins/possible, possible

def horse_avg_pos(past_info: pd.DataFrame, moving_window_size: int) -> tuple[float, int] | tuple[None, None]:
    possible = min(moving_window_size, past_info.shape[0])
    if possible == 0:
        return None, None
    needed_timeframe = past_info.tail(possible)
    mean = needed_timeframe["place"].mean()
    return mean, possible

def horse_avg_lbw(past_info: pd.DataFrame, moving_window_size: int) -> tuple[float, int] | tuple[None, None]:
    possible = min(moving_window_size, past_info.shape[0])
    if possible == 0:
        return None, None
    needed_timeframe = past_info.tail(possible)
    mean = needed_timeframe["LBW"].mean()
    return mean, possible

def jockey_trainer_winrate(past_info: pd.DataFrame, moving_window_size: int) -> float | None:
    possible = min(moving_window_size, past_info.shape[0])
    if possible == 0:
        return None
    needed_timeframe = past_info.tail(possible)
    wins = needed_timeframe[needed_timeframe["place"] == 1].shape[0]
    return wins/possible

def race_time_gap(past_info: pd.DataFrame, race_id: str) -> float | None:
    if past_info.shape[0] == 0:
        return None
    cur_date_format = race_id.split("_")[0]
    last_id = past_info.tail(1)["id"].item()
    last_date_format = last_id.split("_")[0]
    cur_date = date.fromisoformat(cur_date_format)
    last_date = date.fromisoformat(last_date_format)
    diff = (cur_date - last_date).days

    return diff

def horse_avg_class(past_info: pd.DataFrame, moving_window_size: int) -> float | None:
    possible = min(moving_window_size, past_info.shape[0])
    if possible == 0:
        return None
    needed_timeframe = past_info.tail(possible)
    classes = needed_timeframe["class"]
    mean = classes.mean()

    return mean

def avg_sec_quantile(past_result_info_horse: pd.DataFrame, race_id_info: dict) -> list:
    possible = past_result_info_horse.shape[0]
    if possible == 0:
        return [None]*6
    avg_quantiles = [0.0]*6
    contributing_races = [0]*6
    for race_id in past_result_info_horse['id']:
        race_size = race_id_info[race_id].shape[0]
        race_row = past_result_info_horse[past_result_info_horse['id'] == race_id]
        for section in range(6):
            position_at_sec = race_row[f'section_{section+1}'].item()
            if pd.isna(position_at_sec):
                continue
            contributing_races[section] += 1
            avg_quantiles[section] += position_at_sec / race_size
    for section in range(6):
        if contributing_races[section] == 0:
            avg_quantiles[section] = np.nan
            continue
        avg_quantiles[section] = 1- avg_quantiles[section]/contributing_races[section]

    return avg_quantiles

def create_features() -> pd.DataFrame:
    race_info = pd.read_csv("src/data/cleaned_data/cleaned_races.csv")
    result_info = pd.read_csv("src/data/cleaned_data/cleaned_results.csv")

    rows = []

    race_info = race_info.sort_values(by="id", ascending=True).reset_index(drop=True)
    result_info = result_info.sort_values(by="id", ascending=True).reset_index(drop=True)

    results_by_horse = dict(tuple(result_info.groupby('name')))
    results_by_jockey = dict(tuple(result_info.groupby('jockey')))
    results_by_trainer = dict(tuple(result_info.groupby('trainer')))
    results_by_race_id = dict(tuple(result_info.groupby('id')))

    races_by_distance = dict(tuple(race_info.groupby('distance')))
    races_by_going = dict(tuple(race_info.groupby('going')))
    races_by_course = dict(tuple(race_info.groupby('course')))
    races_by_id = dict(tuple(race_info.groupby('id')))
    for index in race_info['id']:

        venue = races_by_id[index]['venue'].item()
        race_class = races_by_id[index]['class'].item()
        going = races_by_id[index]['going'].item()
        distance = races_by_id[index]['distance'].item()
        course = races_by_id[index]['course'].item()

        this_race = results_by_race_id[index]
        available_race_info = race_info[race_info['id'] < index]

        prob_norm_const = 0
        for odd in this_race['odds']:
            prob_norm_const += 1/odd

        #print("normalization constant: ", 1/implied_prob_constant)

        for participant in this_race['name']:
            #info for this race
            info_horse = results_by_horse[participant]
            this_horses_race = info_horse[info_horse['id'] == index]
            jockey = this_horses_race['jockey'].item()
            trainer = this_horses_race['trainer'].item()
            carried_weight = this_horses_race['carried_weight'].item()
            horse_weight = this_horses_race['horse_weight'].item()
            norm_draw = this_horses_race['draw'].item() / this_race['name'].count()
            implied_win_prob = (1 / this_horses_race['odds'].item()) * (1 / prob_norm_const)

            #info for from past races
            hyperparameter_p1 = 1
            hyperparameter_p2 = 5
            hyperparameter_p3 = 5
            hyperparameter_p4 = 1
            hyperparameter_p5 = 1
            hyperparameter_p6 = 1

            available_info_horse = info_horse[info_horse['id'] < index]
            participant_win_rate, participant_win_rate_count = horse_winrate(available_info_horse, hyperparameter_p1)
            participant_avg_pos, participant_avg_pos_count = horse_avg_pos(available_info_horse, hyperparameter_p2)
            d_since_last_race = race_time_gap(available_info_horse, index)
            avg_lbw, lbw_count = horse_avg_lbw(available_info_horse, hyperparameter_p3)


            info_jockey = results_by_jockey[jockey]
            available_info_jockey = info_jockey[info_jockey['id'] < index]
            jockey_win_rate = jockey_trainer_winrate(available_info_jockey,hyperparameter_p4)

            info_trainer = results_by_trainer[trainer]
            available_info_trainer = info_trainer[info_trainer['id'] < index]
            trainer_win_rate = jockey_trainer_winrate(available_info_trainer,hyperparameter_p5)

            # conditioned win rates
            races_with_distance = races_by_distance[distance]
            horse_with_distance = available_info_horse[available_info_horse['id'].isin(races_with_distance['id'])]
            distance_avg_pos, distance_avg_pos_count = horse_avg_pos(horse_with_distance, len(horse_with_distance))

            races_with_going = races_by_going[going]
            horse_with_going = available_info_horse[available_info_horse['id'].isin(races_with_going['id'])]
            going_avg_pos, going_avg_pos_count = horse_avg_pos(horse_with_going, len(horse_with_going))

            races_with_course = races_by_course[course]
            horse_with_course = available_info_horse[available_info_horse['id'].isin(races_with_course['id'])]
            course_avg_pos, course_avg_pos_count = horse_avg_pos(horse_with_course, len(horse_with_course))

            available_races_with_horse = available_race_info[available_race_info['id'].isin(available_info_horse['id'])]
            avg_class = horse_avg_class(available_races_with_horse, hyperparameter_p6)

            (avg_section_pos_1, avg_section_pos_2, avg_section_pos_3, avg_section_pos_4, avg_section_pos_5,
             avg_section_pos_6) = avg_sec_quantile(available_info_horse, results_by_race_id)

            label = 1 if this_horses_race['place'].item() == 1 else 0


            new_row = {
                "id": index,
                "horse": participant,
                "venue": venue,
                "going": going,
                "course": course,
                "jockey": jockey,
                "trainer": trainer,
                "race class": race_class,
                "distance": distance,
                "carried weight": carried_weight,
                "horse weight": horse_weight,
                "normalized draw": norm_draw,
                "implied win probability": implied_win_prob,
                "horses winrate": participant_win_rate,
                "horses average position": participant_avg_pos,
                "days since last race": d_since_last_race,
                "average lbw": avg_lbw,
                "jockey winrate": jockey_win_rate,
                "trainer winrate": trainer_win_rate,
                "average position for race distance": distance_avg_pos,
                "average position for race going": going_avg_pos,
                "average position for race course": course_avg_pos,
                "average class": avg_class,
                "average quantile at section 1": avg_section_pos_1,
                "average quantile at section 2": avg_section_pos_2,
                "average quantile at section 3": avg_section_pos_3,
                "average quantile at section 4": avg_section_pos_4,
                "average quantile at section 5": avg_section_pos_5,
                "average quantile at section 6": avg_section_pos_6,
                "label": label,
                #needed for later calculation and not as a feature
                "prob_norm_const": prob_norm_const,
                "odds": this_horses_race['odds'].item()
            }
            rows.append(new_row)
    return pd.DataFrame(rows)
