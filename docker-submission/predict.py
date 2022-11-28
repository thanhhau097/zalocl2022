import os
import json
import pandas as pd

import subprocess

data_folder = os.path.abspath("/data")

subprocess.run(f"cd ./hoanganh && python predict.py --data_folder {data_folder}", shell=True)
subprocess.run(f"cd ./hoanganh-mergetokens && python predict.py --data_folder {data_folder}", shell=True)


# ensemble
submission_folders = []
for folder in os.listdir("./hoanganh/data/"):
    if os.path.isdir("./hoanganh/data/" + folder):
        submission_folders.append("./hoanganh/data/" + folder)
        
for folder in os.listdir("./hoanganh-mergetokens/data/"):
    if os.path.isdir("./hoanganh-mergetokens/data/" + folder):
        submission_folders.append("./hoanganh-mergetokens/data/" + folder)

ensemble_submission_folder = "ensembled/"
os.makedirs(ensemble_submission_folder, exist_ok=True)
vote_lim = 2

for file in os.listdir(submission_folders[0]):
    datas = []
    for i in range(len(submission_folders)):
        with open(os.path.join(submission_folders[i], file), "r") as f:
            data = json.load(f)
        datas.append(data)

    words_folds = []
    for i in range(len(submission_folders)):
        words = []
        for segment in datas[i]:
            words += segment["l"]
        words_folds.append(words)


    for i in range(len(words_folds[0])):
        words_folds[0][i]["s"] = sorted([words_folds[k][i]["s"] for k in range(len(submission_folders))])[vote_lim]

    for i in range(len(words) - 1):
        words_folds[0][i]["e"] = sorted([words_folds[k][i]["e"] for k in range(len(submission_folders))])[::-1][vote_lim]


    with open(os.path.join(ensemble_submission_folder, file), "w") as f:
        json.dump(datas[0], f)


# tune
submission_folder = ensemble_submission_folder
tune_submission_folder = "./submissions/"
os.makedirs(tune_submission_folder, exist_ok=True)

for file in os.listdir(submission_folder):
    with open(os.path.join(submission_folder, file), "r") as f:
        data = json.load(f)
    words = []

    for segment in data:
        words += segment["l"]

    for i in range(len(words)):
        words[i]["s"] = words[i]["s"] - 20 if words[i]["s"] - 20 > 0 else 0
        words[i]["e"] = words[i]["e"] + 10

    for i in range(len(words) - 1):
        if words[i + 1]["s"] > words[i]["e"]:
            words[i]["e"] = words[i+1]["s"] if words[i+1]["s"] - words[i]["e"] < 500 else words[i]["e"] + 500

    with open(os.path.join(tune_submission_folder, file), "w") as f:
        json.dump(data, f)

# sum time and save to csv
with open("./hoanganh/data/time_submission.json", "r") as f:
    time_dict_1 = json.load(f)
with open("./hoanganh-mergetokens/data/time_submission.json", "r") as f:
    time_dict_2 = json.load(f)

for k, v in time_dict_1.items():
    time_dict_1[k] = v + time_dict_2[k]


df = pd.DataFrame({"fname": list(time_dict_1.keys()), "time": list(time_dict_1.values())})
df.to_csv("time_submission.csv")

os.makedirs("/result", exist_ok=True)
subprocess.run("zip -qqr /result/submissions.zip ./submissions/", shell=True)