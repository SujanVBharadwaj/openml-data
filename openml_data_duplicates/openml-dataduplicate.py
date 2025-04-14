import arff
from collections import defaultdict
import logging
import openml
import os
import pandas as pd
import sys
import numpy as np
from difflib import SequenceMatcher

def load_arff(file_path):
    with open(file_path, 'r') as fh:
        return arff.load(fh)

def get_metafeatures(data):
    name = data['relation']
    instances = len(data['data'])
    features = len(data['data'][0])
    missing = len([v for row in data['data'] for v in row if v is None])
    return name, instances, features, missing

def is_similar(a, b, threshold=0.8):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() > threshold

def get_tags_for_dataset(did):
    try:
        return openml.datasets.get_dataset(did, download_data=False).tags
    except:
        return set()

def compare(data_features, characteristics):
    name, instances, features, missing = data_features
    name_match = is_similar(name, characteristics['name'])
    return (
        name_match,
        abs(characteristics.get('NumberOfInstances', float('nan')) - instances),
        abs(characteristics.get('NumberOfFeatures', float('nan')) - features),
        abs(characteristics.get('NumberOfMissingValues', float('nan')) - missing),
        name  # return original name too for debugging
    )

def create_df_matches(datasets_check=None):
    oml_datasets = openml.datasets.list_datasets()
    comparisons = []
    new_datasets = openml.datasets.get_datasets(datasets_check, download_data=False)

    for i, data in enumerate(new_datasets):
        file_path = data.id
        logging.info("[{:3d}/{:3d}] {}".format(i+1, len(datasets_check), data.name))

        new_data = oml_datasets.get(file_path)
        new_data_metafeatures = [new_data.get(k) for k in ('name','NumberOfInstances','NumberOfFeatures','NumberOfMissingValues')]
        tags_a = set(get_tags_for_dataset(data.id))

        for did, oml_dataset in oml_datasets.items():
            if did == data.id:
                continue
            name_match, d_instances, d_features, d_missing, _ = compare(new_data_metafeatures, oml_dataset)
            tags_b = set(get_tags_for_dataset(did))
            wrong_tags = not bool(tags_a & tags_b)
            if name_match or sum([d_instances, d_features, d_missing]) == 0:
                comparisons.append([
                    data.id, data.name, oml_dataset.get('name'), did,
                    name_match, d_instances, d_features, d_missing, wrong_tags
                ])
    return pd.DataFrame(comparisons, columns=['did', 'name', 'name_duplicate', 'did_duplicate', 'name_match', 'd_instances', 'd_features', 'd_missing', 'wrong_tags'])

def get_matches_per_dataset(df, fn, exclude=[]):
    matches = defaultdict(list)
    for i, row in df.iterrows():
        if row['did_duplicate'] in exclude:
            continue
        if fn(row):
            matches[row['name']].append(row['did_duplicate'])
    return matches

def combine_lists(lists):
    return [y for x in lists.values() for y in x]

def row_print_dict(d, df):
    if len(d) == 0:
        print("[empty]")
        return
    max_len = max([len(k) for k in d])
    for k, values in d.items():
        print('({}){}: {}'.format(int(df[df.name==k].did[:1]), k.ljust(max_len), values))

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig()

    logging.info("Checking for matches against OpenML.")

    source_datasets = openml.study.get_suite(271).data
    df = create_df_matches(source_datasets)

    matched_datasets = []

    def perfect_match(row): 
        return row['name_match'] and row['d_instances'] == 0 and row['d_features'] == 0 and row['d_missing'] == 0   

    perfect_matches = get_matches_per_dataset(df, fn=perfect_match, exclude=matched_datasets)
    matched_datasets += combine_lists(perfect_matches)
    print("Perfect matches:")
    row_print_dict(perfect_matches, df)

    def close_match(row):
        return row['name_match'] and sum([row['d_instances'] == 0, row['d_features'] == 0, row['d_missing'] == 0]) == 2

    close_matches = get_matches_per_dataset(df, fn=close_match, exclude=matched_datasets)
    matched_datasets += combine_lists(close_matches)
    print("Close matches:")
    row_print_dict(close_matches, df)

    def name_match(row):
        return row['name_match'] and sum([row['d_instances'] == 0, row['d_features'] == 0, row['d_missing'] == 0]) < 2

    name_matches = get_matches_per_dataset(df, fn=name_match, exclude=matched_datasets)
    matched_datasets += combine_lists(name_matches)
    print("Name matches:")
    row_print_dict(name_matches, df)

    def shape_match(row):
        return (not row['name_match']) and row['d_instances'] == 0 and row['d_features'] == 0 and row['d_missing'] == 0

    shape_matches = get_matches_per_dataset(df, fn=shape_match)
    matched_datasets += combine_lists(shape_matches)
    print("Shape-only matches:")
    row_print_dict(shape_matches, df)

    print("Datasets with tag mismatches:")
    tag_mismatches = df[df['wrong_tags'] == True]
    print(tag_mismatches[['did', 'name', 'did_duplicate', 'name_duplicate']])

    all_datasets = df['did']
    no_matches = [did for did in all_datasets if did not in matched_datasets]
    print("No matches:")
    for no_match in no_matches:
        print(no_match)
