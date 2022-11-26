import os
import json

# submission_folder = "./data/train/labels_2/"
# tune_submission_folder = "./data/train/labels_4/"
# submission_folder = "./data/spotify_segment_pseudo_result_3/"
# tune_submission_folder = "./data/spotify_segment_pseudo_result_4/"

# submission_folder = "/Users/anhph/Desktop/submissions/"
# tune_submission_folder = "/Users/anhph/Desktop/tuned/"

submission_folder = "data/ensembled/"
tune_submission_folder = "data/tuned/"

for file in os.listdir(submission_folder):
    # try:
        with open(os.path.join(submission_folder, file), "r") as f:
            data = json.load(f)
        words = []
        
        for segment in data:
            words += segment["l"]
#             for word in segment["l"]:
#                 print(word)
            
#             print("-----")
            
        
        for i in range(len(words)):
            words[i]["s"] = words[i]["s"] - 0 if words[i]["s"] - 0 > 0 else 0

        for i in range(len(words) - 1):
            # if words[i + 1]["s"] < words[i]["e"]:
            #     words[i]["e"] = words[i + 1]["s"] + 0
            # else:
            #     words[i]["e"] += 0
            if words[i + 1]["s"] > words[i]["e"]:
                words[i]["e"] = words[i+1]["s"] if words[i+1]["s"] - words[i]["e"] < 500 else words[i]["e"] + 500
            # else:
            #     # INPROGRESS: if the end is too big, should minus it using the start of next word -> NO
            #     if words[i]["e"] - words[i + 1]["s"] > 50:
            #         words[i]["e"] = words[i + 1]["s"] + 50
                    
        
        for word in words:
            print(word)
        # print(data[-1]["e"])
        print("------------------------------------------------------------------")
        with open(os.path.join(tune_submission_folder, file), "w") as f:
            json.dump(data, f)
    # except:
    #     continue