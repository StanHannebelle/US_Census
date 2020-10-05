import pandas as pd
from utils.config import COLUMN_NAMES


def load_and_prep(path):

    data = open_csv(path)

    data['education'] = data['education'].apply(education_encoding)
    data['is_woman'] = data['sex'].apply(is_woman_encoding)
    data['has_private_job'] = data['class_of_worker'].apply(
        has_private_job_encoding)
    data['is_high_wage_sector'] = data['major_industry_code'].apply(
        is_high_wage_sector_encoding)
    data['is_high_wage_occupation'] = data['major_occupation_code'].apply(
        is_high_wage_occupation_encoding)
    data['is_self_employed'] = data.apply(lambda x: is_self_employed_encoding(
        x.class_of_worker, x.own_business_or_self_employed), axis=1)
    data['is_full_time_worker'] = data.apply(lambda x: is_full_time_worker_encoding(
        x.full_or_part_time_employment_stat, x.weeks_worked_in_year), axis=1)
    data['has_capital'] = data.apply(lambda x: has_capital_encoding(
        x.capital_gains, x.capital_losses, x.divdends_from_stocks), axis=1)
    data['is_householder'] = data.detailed_household_summary_in_household.apply(
        is_householder_encoding)
    data['is_married'] = data.marital_status.apply(is_married_encoding)
    data['is_veteran'] = data.veterans_benefits.apply(is_veteran_encoding)
    data['label'] = data.label.apply(label_encoding)

    selected_features = ['age', 'education', 'is_woman', 'has_private_job', 'is_high_wage_sector',
                         'is_high_wage_occupation', 'is_self_employed', 'is_full_time_worker', 'has_capital', 'is_householder', 'is_married', 'is_veteran', 'label']
    data = data[selected_features]

    return(data.values[:, :-1], data.values[:, -1])


def open_csv(path):
    return(pd.read_csv(path, names=COLUMN_NAMES))


def education_encoding(education_string):
    if education_string == ' Children':
        return(0)
    elif education_string in [' Less than 1st grade', ' 1st 2nd 3rd or 4th grade', ' 5th or 6th grade', ' 7th and 8th grade', ' 9th grade', ' 10th grade', ' 11th grade', ' 12th grade no diploma']:
        return(1)
    elif education_string in [' High school graduate', ' Some college but no degree']:
        return(2)
    elif education_string in [' Associates degree-occup /vocational', ' Associates degree-academic program', ' Prof school degree (MD DDS DVM LLB JD)']:
        return(3)
    elif education_string in [' Bachelors degree(BA AB BS)']:
        return(4)
    elif education_string in [' Masters degree(MA MS MEng MEd MSW MBA)']:
        return(5)
    elif education_string in [' Doctorate degree(PhD EdD)']:
        return(6)
    else:
        return(2)


def is_woman_encoding(gender_string):
    if gender_string == ' Female':
        return(1)
    else:
        return(0)


def has_private_job_encoding(class_of_worker_string):
    if class_of_worker_string in [' Private', ' Self-employed-incorporated', ' Self-employed-not incorporated']:
        return(1)
    else:
        return(0)


def is_high_wage_sector_encoding(sector_string):
    if sector_string in [' Manufacturing-durable goods', ' Finance insurance and real estate', ' Public administration', ' Utilities and sanitary services', ' Communications', ' Mining', ' Armed Forces']:
        return(1)
    else:
        return(0)


def is_high_wage_occupation_encoding(occupation_string):
    if occupation_string in [' Professional specialty', ' Executive admin and managerial', ' Sales', ' Protective services', ' Armed Forces']:
        return(1)
    else:
        return(0)


def is_self_employed_encoding(class_of_worker_string, own_business_or_self_employed_string):
    if own_business_or_self_employed_string != 0:
        return(1)
    if class_of_worker_string in [' Self-employed-not incorporated', ' Self-employed-incorporated']:
        return(1)
    else:
        return(0)


def is_full_time_worker_encoding(full_or_part_time_employment_stat_string, weeks_worked_in_year_string):
    if full_or_part_time_employment_stat_string == ' Full-time schedules' and weeks_worked_in_year_string == 52:
        return(1)
    else:
        return(0)


def has_capital_encoding(capital_gains_string, capital_losses_string, divdends_from_stocks_string):
    if capital_gains_string == 0 and capital_losses_string == 0 and divdends_from_stocks_string == 0:
        return(0)
    else:
        return(1)


def is_householder_encoding(household_string):
    if household_string == ' Householder':
        return(1)
    else:
        return(0)


def is_married_encoding(married_string):
    if married_string == ' Never married':
        return(0)
    else:
        return(1)


def is_veteran_encoding(veteran_string):
    if veteran_string == 0:
        return(0)
    else:
        return(1)


def label_encoding(label_string):
    if label_string == ' - 50000.':
        return(0)
    else:
        return(1)
