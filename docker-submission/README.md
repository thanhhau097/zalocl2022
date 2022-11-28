# inference
```
docker run --gpus "device=0" -v /home/thanh/zalocl2022/docker-submission/data:/data -v /home/thanh/zalocl2022/docker-submission/result:/result -it thanhtest /bin/bash /code/predict.sh
```

# jupyter notebook
```
docker run --gpus "device=0" -v /home/thanh/zalocl2022/docker-submission/data:/data -v /home/thanh/zalocl2022/docker-submission/result:/result -p9777:9777 -it thanhtest /bin/bash /code/start_jupyter.sh
```