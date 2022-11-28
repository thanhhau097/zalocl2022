import os
import json

submission_folder = "./data/train/labels/"
tune_submission_folder = "./data/train/labels_2/"


for file in os.listdir(submission_folder):
    try:
        with open(os.path.join(submission_folder, file), "r") as f:
            data = json.load(f)
        
        words = []
        
        for segment in data:
            words += segment["l"]
            
        for i in range(len(words)):
            words[i]["s"] = words[i]["s"] - 50 if words[i]["s"] - 50 > 0 else 0

        for i in range(len(words) - 1):
            if words[i + 1]["s"] < words[i]["e"]:
                words[i]["e"] = words[i + 1]["s"] + 0
            else:
                words[i]["e"] += 0

        with open(os.path.join(tune_submission_folder, file), "w") as f:
            json.dump(data, f)
    except:
        continue