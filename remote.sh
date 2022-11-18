#!/bin/bash
set -x

TEMP_FILE=.tmp
PROJECT_NAME=$(basename "`pwd`")
echo "Run training for project $PROJECT_NAME"

/bin/cat <<EOM >$TEMP_FILE
.vscode/*
.dvc/cache/*
.git/*
*.nii.gz
lib_ds.egg-info
logs/*
outputs/*
weights/*
notebooks/*
*.ipynb
*.whl
*.bin
*.cpp
*.jpg
*.o
*.so
*.flac
*.mp4
*.jpg
*.png
*.tar
*.dcm
*.mp3
*.dicom
*.xml
*.swp
*.pb
*.tmp
*.csv
data/train/
data/test/
data/index/
*.onnx
*.zip
*.parquet
*.html
*.feather
*.pth
*.npy.gz
*.wav
*.pkl
*.npy
*.jpg
*.zip
*.log
*.tiff
*.wav
lib_ds/models/db/trainingv2/task_91.json
EOM

if [ "$1" == "multi" ]; then
    echo "Push code to multi-202201071427.us-central1-a.neural2-prod-2"
    IP="multi-202201071427.us-central1-a.neural2-prod-2"
    REMOTE_HOME="/root"
elif [ "$1" == "matpat" ]; then
    echo "Push code to matpatclassify.us-central1-a.neural2-prod-2"
    IP="matpatclassify.us-central1-a.neural2-prod-2"
    REMOTE_HOME="/root"
elif [ "$1" == "sprout" ]; then
    echo "Push code to sprout server"
    IP="sprout"
    REMOTE_HOME="/home/nhan"
elif [ "$1" == "coach" ]; then
    echo "Push code to coach"
    IP="coach"
    REMOTE_HOME="/storage"
elif [ "$1" == "lambda" ]; then
    echo "Push code to lambdalab"
    IP="ubuntu@150.136.116.51"
    REMOTE_HOME="/home/ubuntu"
else
    echo "Unknown instance"
    exit
fi

# push code to server
rsync -vr -P -e "ssh" --exclude-from $TEMP_FILE "$PWD" $IP:$REMOTE_HOME/

# remove temp. file
rm $TEMP_FILE
