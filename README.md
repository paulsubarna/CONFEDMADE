
# Masked Autoencoders are Efficient Continual Federated Learners

This repository is an official Tensorflow 2 implementation of [Masked Autoencoders are Efficient Continual Federated Learners] (**NeurIps 2023**) 



The main contributions of this work are as follows:

* We draw inspiration from the supervised FedWeIT and extend it to our unsupervised **Con**tinual **Fed**erated **MA**sked autoencoders for **D**ensity **E**stimation (**CONFEDMADE**); an unsupervised continual federated learner based on masking to enable selective knowledge transfer between clients and reduce forgetting.* Through our intelligent masking strategy, we are still successful in achieving desirable performances even after sparsifying the model parameters by 70 %. 
* We highlight that MADE is a model particularly amenable to CFL and investigate several non-trivial considerations, such as connectivity and masking strategy, beyond a trivial application of federated averaging and FedWeIT to the unsupervised setting.
* We extensively evaluate our approach on several CFL scenarios on both image and numerical data. Overall, CONFEDMADE consistently reduces forgetting while sparsifying parameters and reducing communication costs with respect to a naive unsupervised CFL approach.

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
Run below script to generate datasets. 
The --task parameter has three choices: `mnist`, `bianry`, and `non_miid (Mnist+ Emnist)` to generate the desired type of task sets for the clients.

```bash
python3 ../main.py --work-type gen_data --task mnist --seed 777 
```  

## Run Experiments
To reproduce experiments, please execute `train_mnist.sh` file in the `scripts` folder, or you may run the following comamnd line directly:

```bash
python3 ../main.py --gpu 0,1,2 \
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


