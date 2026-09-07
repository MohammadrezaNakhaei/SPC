import numpy as np 
import torch 
import pathlib, h5py
from collections import deque
from torch.utils.data import IterableDataset, DataLoader

class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_dim, action_dim, capacity, device, verbose=True):
        assert isinstance(obs_dim, int) and isinstance(action_dim, int)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.capacity = capacity
        total_shape = 2*self.obs_dim + self.action_dim + 2
        total_bytes = 4*total_shape*capacity # 4 bytes for each tensor
        if verbose: print(f'Storage required: {total_bytes/1e9:.2f} GB') 
        storage_device = 'cpu' # by default
        if 'cuda' in device:
            mem_free, _ = torch.cuda.mem_get_info()
            # Heuristic: decide whether to use CUDA or CPU memory
            storage_device = 'cuda:0' if 2.5*total_bytes < mem_free else 'cpu'
        if verbose: print(f'Using {storage_device.upper()} memory for storage.')
        self.device = device
        self.storage_device = storage_device
        self.tensors = torch.empty((capacity, total_shape), dtype=torch.float32, device=storage_device)
        self.idx = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done,):
        self.tensors[self.idx] = torch.as_tensor(
            np.concatenate([obs, action, [reward], next_obs, [done]]), 
            dtype=torch.float32)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0
        
    def add_torch(self, obs, action, reward, next_obs, done):
        for tensor in obs, action, reward, next_obs, done:
            assert tensor.ndim==2 
        batch_size = obs.shape[0]
        idxs = torch.arange(self.idx, self.idx+batch_size, device=self.storage_device)%self.capacity
        self.idx = (self.idx+batch_size)%self.capacity
        self.full = self.full or self.idx+batch_size>self.capacity
        self.tensors[idxs] = torch.cat([obs, action, reward, next_obs, done], dim=-1).to(self.storage_device) 

    def sample(self, batch_size, return_next_action=False):
        idxs = torch.randint(0, self.capacity if self.full else self.idx, size=(batch_size,), device=self.storage_device)
        tensor = self.tensors[idxs]
        if self.device=='cuda:0' and self.storage_device =='cpu':
            tensor = tensor.to('cuda:0')
        obss = tensor[:, :self.obs_dim]
        actions = tensor[:, self.obs_dim:self.obs_dim+self.action_dim]
        rewards = tensor[:, self.obs_dim+self.action_dim: self.obs_dim+self.action_dim+1]
        next_obss = tensor[:, -self.obs_dim-1:-1]
        dones = tensor[:, -1:]
        if return_next_action:
            next_actions = self.tensors[(idxs+1)%self.capacity, self.obs_dim:self.obs_dim+self.action_dim]
            return obss, actions, rewards, next_obss, dones, next_actions
        return obss, actions, rewards, next_obss, dones
    
    def save(self, path):
        torch.save(self.tensors[:len(self)].to('cpu'), path)

    def load(self, path):
        data = torch.load(path)
        assert data.shape[0]<=self.capacity, 'buffer is too small for the dataset'
        assert data.shape[1]==self.tensors.shape[1], 'buffer shape does not match the dataset'
        self.idx = data.shape[0]
        self.tensors[:self.idx] = data
        self.full = self.idx==self.capacity

    def load_d4rl(self, data):
        assert data['rewards'].shape[0]<=self.capacity, 'buffer is too small for the dataset'
        tensor = torch.as_tensor(
                    np.concatenate([data['observations'], data['actions'], data['rewards'][:, None], data['next_observations'], data['terminals'][:, None]], axis=1), 
                    dtype=torch.float32)
        self.tensors = tensor.to(self.storage_device) 
        self.idx = tensor.shape[0]


class SeqBuffer():
    """Sample a sequenctial batch of data"""
    def __init__(self, buffer:ReplayBuffer, seq_len, ep_length):
        assert len(buffer)>0
        self.buffer = buffer 
        self.seq_len = seq_len
        self.ep_length = ep_length
        self.obs_dim = self.buffer.obs_dim
        self.action_dim = self.buffer.action_dim

        self.valid_idx = []

        ind = torch.where(buffer.tensors[:, -1]==1)[0] # dones
        init_ep = (ind+1).cpu().numpy().tolist()
        init_ep.insert(0, 0) # assuming the first transition is the initial state, not overridden
        i = 0
        while i < len(init_ep) - 1:
            if init_ep[i + 1] - init_ep[i] > ep_length:
                init_ep.insert(i + 1, init_ep[i] + ep_length)
            self.valid_idx.extend(range(init_ep[i], init_ep[i+1]-self.seq_len+1))
            i += 1
        while init_ep[-1]<len(buffer):
            init_ep.append(init_ep[-1]+ep_length)
            end = min(init_ep[-1]+ep_length-self.seq_len+1, len(buffer)-self.seq_len)
            self.valid_idx.extend(range(init_ep[-1], end))
        self.valid_idx = np.array(self.valid_idx, dtype=np.int64)

    def _sample_single(self, idx):
        return self.buffer.tensors[idx]
    
    def sample(self, batch_size:int):
        idx = np.random.choice(self.valid_idx, size=(batch_size,))
        total_idx = np.vstack([idx+i for i in range(self.seq_len)])
        total_idx = torch.as_tensor(total_idx, device=self.buffer.storage_device)
        tensor = torch.vmap(self._sample_single)(total_idx)
        if self.buffer.device=='cuda:0' and self.buffer.storage_device =='cpu':
            tensor = tensor.to('cuda:0')
        obses = tensor[..., :self.obs_dim]
        actions = tensor[..., self.obs_dim:self.obs_dim+self.action_dim]
        rewards = tensor[..., self.obs_dim+self.action_dim: self.obs_dim+self.action_dim+1]
        next_obses = tensor[..., -self.obs_dim-1:-1]
        dones = tensor[..., -1:]
        return obses, actions, rewards, next_obses, dones


class MultiReplayBuffer():
    """Buffer storing samples of similar tasks"""
    def __init__(self, buffer_idxs, obs_dim, action_dim, capacity, device):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.capacity = capacity
        total_shape = 2*self.obs_dim + self.action_dim + 2
        total_bytes = len(buffer_idxs)*4*total_shape*capacity # 4 bytes for each tensor
        print(f'Storage required: {total_bytes/1e9:.2f} GB') 
        storage_device = 'cpu' # by default
        if 'cuda' in device:
            mem_free, _ = torch.cuda.mem_get_info()
            # Heuristic: decide whether to use CUDA or CPU memory
            storage_device = 'cuda:0' if 2.5*total_bytes < mem_free else 'cpu'
        print(f'Using {storage_device.upper()} memory for storage.')
        self._buffers = {
            idx: ReplayBuffer(obs_dim, action_dim, capacity, storage_device, verbose=False) 
            for idx in buffer_idxs
            }
        self.storage_device = storage_device
        self.device = device

    def sample(self, buffer_idxs, batch_size, return_next_action=False):
        '''
        returns obses, actions, rewards, next_obses, dones, next_action(optional)
        '''
        unpacked = [
            self._buffers[idx].sample(batch_size, return_next_action=return_next_action)
              for idx in buffer_idxs]
        # group like elements together
        unpacked = [[x[i].unsqueeze(0) for x in unpacked] for i in range(len(unpacked[0]))]
        unpacked = [torch.cat(x, dim=0) for x in unpacked]
        if self.storage_device=='cpu' and 'cuda' in self.device:
            unpacked = [x.to(self.device) for x in unpacked]
        return unpacked # obses, actions, rewards, next_obses, dones
    
    def load(self, buffer_idx, path):
        self._buffers[buffer_idx].load(path)


class MultiSequenceBuffer():
    """"""
    def __init__(self, buffer:MultiReplayBuffer, seq_len:int=5, ep_length:int=200):
        self.num_tasks = len(buffer._buffers)
        self.seq_len = seq_len
        self.buffer = buffer
        self._seq_buffers = {
            i: SeqBuffer(buffer._buffers[i], seq_len, ep_length) for i in range(self.num_tasks)
        }
        self.device = buffer.device
        self.storage_device = buffer.storage_device

    def sample(self, buffer_idxs, batch_size,):
        unpacked = [
            self._seq_buffers[idx].sample(batch_size)
              for idx in buffer_idxs]
        # group like elements together
        unpacked = [[x[i].unsqueeze(0) for x in unpacked] for i in range(len(unpacked[0]))]
        unpacked = [torch.cat(x, dim=0) for x in unpacked]
        if self.storage_device=='cpu' and 'cuda' in self.device:
            unpacked = [x.to(self.device) for x in unpacked]
        return unpacked # obses, actions, rewards, next_obses, dones

