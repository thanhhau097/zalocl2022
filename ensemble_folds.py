import os
import json

# submission_folder = "./data/train/labels_2/"
# tune_submission_folder = "./data/train/labels_4/"
# submission_folder = "./data/spotify_segment_pseudo_result_3/"
# tune_submission_folder = "./data/spotify_segment_pseudo_result_4/"

submission_folders = ["data/submissions_f5/", "data/submissions_f7/", "data/submissions_f9/"]
tune_submission_folder = "data/ensembled/"

for file in os.listdir(submission_folders[0]):
    # try:

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
            words_folds[0][i]["s"] = sorted([words_folds[k][i]["s"] for k in range(len(submission_folders))])[1]

        for i in range(len(words) - 1):
            words_folds[0][i]["e"] = sorted([words_folds[k][i]["e"] for k in range(len(submission_folders))])[1]
                    
        
        # for word in words:
        #     print(word)
        # print(data[-1]["e"])
        # print("------------------------------------------------------------------")
        with open(os.path.join(tune_submission_folder, file), "w") as f:
            json.dump(datas[0], f)
    # except:
    #     continue