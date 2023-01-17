# FCLR
# Learning Representations from Indirect Experience through Unsupervised Continual Federated Learning

This repository is an official Tensorflow 2 implementation of [Learning Representations from Indirect Experience through Unsupervised
Continual Federated Learning ] (**ICML 2023**)



## Abstract

Federated learning is deemed to be particularly useful in real-world scenarios where we collaboratively learn a global model while ensuring data-privacy between the clients. Continual learning can add an extra paradigm to FL as previous works heavily suffer from catastrophic forgetting under different data distributions. With little to no research directed towards combining both concepts, FL frameworks also focused heavily towards classification tasks. Foreseeing the ever-growing problem of data annotation, this work proposes an approach that aims to learn representations both from direct and indirect experiences through unsupervised Federated  Continual Learning. Inspired from a recent work, this work follows up on the idea of bridging the gap between FL and CL methods even further through selective representational knowledge transfer between clients. The novelty of this work lies in the introduction of an intelligent masking technique that can combat the communication overheads in FL while learning representations from indirect experiences. We have empirically evaluated our framework under several Non-IID scenarios using images and binary datasets. We have shown that our proposed approach is successful in achieving desirable results under a highly sparsified network.

The main contributions of this work are as follows:

* We have proposed an **unsupervised Federated Continual Learning approach** with selective knowledge transfer between the clients to reduce catastrophic forgetting. 
* Through our intelligent masking strategy, we are still successful in achieving desirable performances even after sparsifying the model parameters by 70 \%. 
* We have extensively evaluated our approach in several Non-IID scenarios and have shown that it doesn't fall victim to catastrophic forgetting.


## Environmental Setup

Please install packages from `requirements.txt` after creating your own environment with `python 3.8.x`.

```bash
$ pip install --upgrade pip
$ pip install -r requirements.txt
```

## Data Generation
Please see `config.py` to set your custom path for both `datasets` and `output files`.
```python
args.task_path = '/path/to/taskset/'  # for task sets of each client
args.output_path = '/path/to/outputs/' # for logs, weights, etc.
```
Run below script to generate datasets

```bash
python3 ../main.py --work-type gen_data --task mnist --seed 777 
```
The --task parameter has three choices: `mnist`, `bianry`, and `non_miid` to generate the desired type of task sets for the clients.  

## Run Experiments
To reproduce experiments, please execute `train-non-iid-50.sh` file in the `scripts` folder, or you may run the following comamnd line directly:

```bash
python3 ../main.py --gpu 0,1,2,3,4 \
		--work-type train \
		--model fedweit \
		--task non_iid_50 \
	 	--gpu-mem-multiplier 9 \
		--num-rounds 20 \
		--num-epochs 1 \
		--batch-size 100 \
		--seed 777 
```
Please replace arguments as you wish, and for the other options (i.e. hyper-parameters, etc.), please refer to `config.py` file at the project root folder.

> Note: while training, all participating clients are logically swiched across the physical gpus given by `--gpu` options (5 gpus in the above example). 

## Results
All clients and server create their own log files in `\path\to\output\logs\`, which include evaluation results, such as local & global performance and communication costs, and the experimental setups, such as learning rate, batch-size, etc. The log files will be updated for every comunication rounds. 

## Citations
```
@inproceedings{
    yoon2021federated,
    title={Federated Continual Learning with Weighted Inter-client Transfer},
    author={Jaehong Yoon and Wonyong Jeong and Giwoong Lee and Eunho Yang and Sung Ju Hwang},
    booktitle={International Conference on Machine Learning},
    year={2021},
    url={https://arxiv.org/abs/2003.03196}
}
```
