# Temporal Consistency Improves Generalization in Contextual Offline Meta-RL
We study temporal consistency for task representation learning for Offline Meta-RL. Temporal consistency in RL is also known as self-predictive RL, so we use **SPC** for Self-Predictive Contextual OMRL. 
## Instructions 
### Install dependencies:
```sh
conda env create -f environment.yaml
conda activate spc
```
### Collecting datasets:
``` sh
python collect_data.py +env=ant-dir goal_idx=0 use_wandb=true
```
The ```+env``` argeument is compulsory defining the environment listed in  ```cfgs/env``` directory. The ```goal_idx``` is the task id which is between (0, num_tasks) where num_tasks is specified in the coresponding environment config file. By default, we log metrics with wandb. You can ignore this by setting ```use_wandb=false```. 
The datasets are stored in ```data/``` directory.

Also you can speed up the training by setting ``` agent.compile=true``` and ``` agent.cuda_graph=true ```.

We provide datasets for some environments in this [link](https://doi.org/10.5281/zenodo.17189868). Make sure the dataset for each environment is in ```data/{env_name}``` directory.

### Training OMRL agents (baselines)
``` sh
python train_omrl.py +env=ant-dir +agent=unicorn use_wandb=true seed=0
```
Please note that you need to collect datasets for all of the tasks (```goal_idx=0:num_tasks```). 
The ```+agent``` and ```+env``` argeuments are compulsory. 
By default, metrics for each experiments are logged with wandb and default seed is 0. 
You can see the OMRL methods in ```cfgs/agents``` and by changing ```+agents``` argeument, you can train different offline meta RL agents. The agent is saved at ```logs/``` directory. 

You can see all the hyperparameters in ```cfgs/omrl.yaml``` and ```cfgs/agent/common.yaml```.

```train_omrl``` is a vectorized implementation for evaluation, ```train_omrl_single_env``` does the same without vectorzation (will be slower).

### Training contextual world models
``` sh
python train_spc.py +env=ant-dir agent.world_model=continuous_mse use_wandb=true seed=0
```
Please note that you need to collect datasets for all of the tasks (```goal_idx=0:num_tasks```). 
the ```+env``` argeument is compulsory. ```agent.world_model``` defines the latent formulation (e.g., discrete_ce is discrete with cross entropy, ```continuous_mse``` is continuous with MSE loss function, etc). The default value is ```discrete_ce```. 
You can see all the hyperparameters in ```cfgs/spc.yaml```. 
