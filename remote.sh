#!/bin/bash
set -x

TEMP_FILE=.tmp
PROJECT_NAME=$(basename "`pwd`")
echo "Run training for project $PROJECT_NAME"

/bin/cat <<EOM >$TEMP_FILE
.git/*
logs/*
outputs/*
weights/*
notebooks/*
*.cpp
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
*.ipynb
*.feather
*.pth
*.wav
*.pkl
*.npy
*.jpg
*.zip
EOM

# if [ "$1" == "t4" ]; then
#     echo "Push code to h2-data-sekisan-tokyo-experiment-instance-1"
#     IP="h2-data-sekisan-tokyo-experiment-instance-1.asia-northeast1-a.ai-sekisan-dev"
#     REMOTE_HOME="/home/anhph"
# elif [ "$1" == "cvat" ]; then
#     echo "Push code to h2-data-cvat-tokyo-dev-instance-1"
#     IP="h2-data-cvat-tokyo-dev-instance-1.asia-northeast1-b.ai-sekisan-dev"
#     REMOTE_HOME="/home/anhph"
# elif [ "$1" == "a100" ]; then
#     echo "h2-data-sekisan-tokyo-a100x8-instance-1"
#     IP="h2-data-sekisan-tokyo-a100x8-instance-1.asia-northeast1-a.ai-sekisan-dev"
#     REMOTE_HOME="/home/anhph"
# else
#     echo "Unknown instance"
#     exit
# fi


IP=$1
echo "Push code to $IP"
REMOTE_HOME="/home/phamhoan"

# push code to server
rsync -vr -P -e "ssh" --exclude-from $TEMP_FILE "$PWD" $IP:$REMOTE_HOME/

# remove temp. file
rm $TEMP_FILE
