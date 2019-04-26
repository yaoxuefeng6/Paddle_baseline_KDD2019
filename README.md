# Paddle_baseline_KDD2019
Paddle baseline for KDD2019 "Context-Aware Multi-Modal Transportation Recommendation"(https://dianshi.baidu.com/competition/29/question)

This repository is the demo codes for the  KDD2019 "Context-Aware Multi-Modal Transportation Recommendation" competition using PaddlePaddle. It is writen by python and uses PaddlePaddle to solove the task. Note that this repisitory is on developing and welcome everyone to contribute. The current baseline solution codes can get 0.68 - 0.69 score of online submission. 
The reason of the publication of this baseline codes is to encourage us to use PaddlePaddle and build the most powerful recommendation model via PaddlePaddle.

Note that the code is rough and for my daily use. They will be trimmed these days...
## Install PaddlePaddle
please visit to the official site of PaddlePaddle(http://www.paddlepaddle.org/documentation/docs/zh/1.4/beginners_guide/install/index_cn.html) 
## preprocess feature
preprocess.py and preprocess_dense.py is the code for preprocessing the raw. the two versions are provided to dealing with all sparse feature and sparse plus dense feature. Correspondly, pre_process_test.py and pre_test_dense.py are the codes to preproccess test raw data. 

## build the network
main network logic is in network_confv?.py. I try sereval networks and public some of them. There may be some mistakes on the networks but all of them are functional. 

## train the network
In local_train.py and map_reader.py, I use dataset API, so we need to download the correspond .whl package or clone codes on develop branch of PaddlePaddle

## test results
In generate_test.py and build_submit, for convenience, I use the whole train data to train the network and test the network with provided data without label



