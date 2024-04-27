import json

import pandas as pd
from glob import glob
import os

def create_csv_given_json_answer(out_csv_path: str, json_answer_glob_dir: str, obey_data_path: str = "data/child-obey-scores.txt", materialistic_data_path: str = "data/materialistic-scores.txt"):
    """
    Create a csv file given json answer files.
    :param out_csv_path: The output csv file path.
    :param json_answer_glob_dir: The glob pattern for json answer file paths.
    :return: None
    """
    json_answer_paths = glob(json_answer_glob_dir)
    all_dfs = []  # List to store individual dataframes

    for json_answer_path in json_answer_paths:
        with open(json_answer_path, "r") as f:
            data = json.load(f)
        # FOR OTHER KEY, +1
        for k, v in data.items():
            if k not in ["child-obey1-q8", "materialistic-q155"]:
                data[k] = v + 1

        # CHANGE child-obey1-q8 AND materialistic-q155
        with open(obey_data_path, "r") as f:
            obey_data = f.readlines()
        with open(materialistic_data_path, "r") as f:
            materialistic_data = f.readlines()
            
        data["child-obey1-q8"] = int(obey_data[data["child-obey1-q8"]])
        data["materialistic-q155"] = int(materialistic_data[data["materialistic-q155"]])
        # Extract the filename without extension to use as column name
        filename = os.path.splitext(os.path.basename(json_answer_path))[0]
        # Convert the data dictionary to a DataFrame
        df = pd.DataFrame(list(data.items()), columns=["Question", filename])
        # Set 'Question' as the index
        df.set_index("Question", inplace=True)
        # Append to the list of dataframes
        all_dfs.append(df)

    # Concatenate all dataframes along the columns
    final_df = pd.concat(all_dfs, axis=1)
    # Save the concatenated dataframe to a CSV file
    final_df.to_csv(out_csv_path)
