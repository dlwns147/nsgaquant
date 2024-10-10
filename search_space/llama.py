import numpy as np
from utils import get_net_info
from tqdm import tqdm
import math

class LlamaSearchSpace:
    def __init__(self, num_blocks, pass_linear_list=[], small_model_bits=2, large_model_bits=4, use_prune=False, config=None, sec_obj='bits', sec_obj_range=[]):
        self.num_blocks = num_blocks  # number of blocks

        self.small_model_bits = small_model_bits
        self.large_model_bits = large_model_bits
        self.q_proj_option = [0, small_model_bits, large_model_bits]
        self.k_proj_option = [0, small_model_bits, large_model_bits]
        self.v_proj_option = [0, small_model_bits, large_model_bits]
        self.o_proj_option = [0, small_model_bits, large_model_bits]
        self.down_proj_option = [0, small_model_bits, large_model_bits]
        self.up_proj_option = [0, small_model_bits, large_model_bits]
        self.gate_proj_option = [0, small_model_bits, large_model_bits]

        self.use_prune = use_prune
        self.pass_linear_list = pass_linear_list
        self.config = config
        
        self.sec_obj = sec_obj
        self.sec_obj_range = sec_obj_range
        assert len(sec_obj_range) == 2, f"sec_obj_range is invalid: {sec_obj_range}"
        assert math.isclose(sec_obj_range[0], sec_obj_range[1]) or sec_obj_range[0] < sec_obj_range[1], f"sec_obj_range is invalid: {sec_obj_range}"
        

    def sample(self, n_samples=1, nb=None, q=None, k=None, v=None, o=None, down=None, up=None, gate=None, pool=[]):
        """ randomly sample a architecture"""
        # nb = self.num_blocks if nb is None else nb
        # ks = self.kernel_size if ks is None else ks
        # e = self.exp_ratio if e is None else e
        # d = self.depth if d is None else d
        # r = self.resolution if r is None else r

        nb = self.num_blocks if nb is None else nb
        q = self.q_proj_option if q is None else q
        k = self.k_proj_option if k is None else k
        v = self.v_proj_option if v is None else v
        o = self.o_proj_option if o is None else o
        down = self.down_proj_option if down is None else down
        up = self.up_proj_option if up is None else up
        gate = self.gate_proj_option if gate is None else gate
        
        data = []
        for n in tqdm(range(n_samples), desc='Sampling'):
            while True:
                prob = np.random.rand(3)
                # q_prob = np.random.rand(len(q))
                q_prob = prob[np.array([np.argwhere(_x == np.array(self.q_proj_option))[0, 0] for _x in q])]
                q_list = np.random.choice(q, size=nb, p=q_prob / q_prob.sum(), replace=True).tolist()

                # k_prob = np.random.rand(len(k))
                k_prob = prob[np.array([np.argwhere(_x == np.array(self.k_proj_option))[0, 0] for _x in k])]
                k_list = np.random.choice(k, size=nb, p=k_prob / k_prob.sum(), replace=True).tolist()
                
                # v_prob = np.random.rand(len(v))
                v_prob = prob[np.array([np.argwhere(_x == np.array(self.v_proj_option))[0, 0] for _x in v])]
                v_list = np.random.choice(v, size=nb, p=v_prob / v_prob.sum(), replace=True).tolist()
                
                # o_prob = np.random.rand(len(o))
                o_prob = prob[np.array([np.argwhere(_x == np.array(self.o_proj_option))[0, 0] for _x in o])]
                o_list = np.random.choice(o, size=nb, p=o_prob / o_prob.sum(), replace=True).tolist()

                # down_prob = np.random.rand(len(down))
                down_prob = prob[np.array([np.argwhere(_x == np.array(self.down_proj_option))[0, 0] for _x in down])]
                down_list = np.random.choice(down, size=nb, p=down_prob / down_prob.sum(), replace=True).tolist()
                
                # up_prob = np.random.rand(len(up))
                up_prob = prob[np.array([np.argwhere(_x == np.array(self.up_proj_option))[0, 0] for _x in up])]
                up_list = np.random.choice(up, size=nb, p=up_prob / up_prob.sum(), replace=True).tolist()

                # gate_prob = np.random.rand(len(gate))
                gate_prob = prob[np.array([np.argwhere(_x == np.array(self.gate_proj_option))[0, 0] for _x in gate])]
                gate_list = np.random.choice(gate, size=nb, p=gate_prob / gate_prob.sum(), replace=True).tolist()

                for pass_linear in self.pass_linear_list:
                    blk, module, linear = pass_linear.split('.')
                    blk = int(blk)
                    if linear == 'q_proj':
                        q_list[blk] = max(self.q_proj_option)
                    if linear == 'k_proj':
                        k_list[blk] = max(self.k_proj_option)
                    if linear == 'v_proj':
                        v_list[blk] = max(self.v_proj_option)
                    if linear == 'o_proj':
                        o_list[blk] = max(self.o_proj_option)
                    if linear == 'down_proj':
                        down_list[blk] = max(self.down_proj_option)
                    if linear == 'up_proj':
                        up_list[blk] = max(self.up_proj_option)
                    if linear == 'gate_proj':
                        gate_list[blk] = max(self.gate_proj_option)
                new_arch = {'self_attn.q_proj': q_list, 'self_attn.k_proj': k_list, 'self_attn.v_proj': v_list, 'self_attn.o_proj': o_list, 'mlp.down_proj': down_list, 'mlp.up_proj': up_list, 'mlp.gate_proj': gate_list}
                complexity = get_net_info(new_arch, self.config)[self.sec_obj]
                # print(f'complexity : {complexity:.3f}')
                if (new_arch not in data) and (new_arch not in pool) and (math.isclose(complexity, self.sec_obj_range[0]) or complexity > self.sec_obj_range[0]) and (math.isclose(complexity, self.sec_obj_range[1]) or complexity < self.sec_obj_range[1]):
                    # print(f'selected arch : {complexity} bits')
                    # print('=' * 20)
                    break
                
            data.append(new_arch)
        return data

    def initialize(self, n_doe, pool=[]):
        # sample one arch with least (lb of hyperparameters) and most complexity (ub of hyperparameters)
        data = []
        if math.isclose(self.sec_obj_range[0], self.small_model_bits):
            data.append(self.sample(q=[self.small_model_bits], k=[self.small_model_bits], v=[self.small_model_bits], o=[self.small_model_bits], down=[self.small_model_bits], up=[self.small_model_bits], gate=[self.small_model_bits])[0])
            n_doe -= 1
        if math.isclose(self.sec_obj_range[1], self.large_model_bits):
            data.append(self.sample(q=[self.large_model_bits], k=[self.large_model_bits], v=[self.large_model_bits], o=[self.large_model_bits], down=[self.large_model_bits], up=[self.large_model_bits], gate=[self.large_model_bits])[0])
            n_doe -= 1
        data.extend(self.sample(n_samples=n_doe, q=[self.small_model_bits, self.large_model_bits], k=[self.small_model_bits, self.large_model_bits], v=[self.small_model_bits, self.large_model_bits], o=[self.small_model_bits, self.large_model_bits], down=[self.small_model_bits, self.large_model_bits], up=[self.small_model_bits, self.large_model_bits], gate=[self.small_model_bits, self.large_model_bits], pool=pool))
        return data


    def encode(self, arch):
        # encode arch ({'q': [0, 2, 4], 'k: , etc}) to integer bit-string [1, 0, 2, 1, ...]
        q_encode = np.array([np.argwhere(_x == np.array(self.q_proj_option))[0, 0] for _x in arch['self_attn.q_proj']]).reshape(-1, 1)
        k_encode = np.array([np.argwhere(_x == np.array(self.k_proj_option))[0, 0] for _x in arch['self_attn.k_proj']]).reshape(-1, 1)
        v_encode = np.array([np.argwhere(_x == np.array(self.v_proj_option))[0, 0] for _x in arch['self_attn.v_proj']]).reshape(-1, 1)
        o_encode = np.array([np.argwhere(_x == np.array(self.o_proj_option))[0, 0] for _x in arch['self_attn.o_proj']]).reshape(-1, 1)
        down_encode = np.array([np.argwhere(_x == np.array(self.down_proj_option))[0, 0] for _x in arch['mlp.down_proj']]).reshape(-1, 1)
        up_encode = np.array([np.argwhere(_x == np.array(self.up_proj_option))[0, 0] for _x in arch['mlp.up_proj']]).reshape(-1, 1)
        gate_encode = np.array([np.argwhere(_x == np.array(self.gate_proj_option))[0, 0] for _x in arch['mlp.gate_proj']]).reshape(-1, 1)

        return np.stack((q_encode, k_encode, v_encode, o_encode, down_encode, up_encode, gate_encode), axis=-1).reshape(-1)

    def decode(self, x):
        # decode integer bit-string [1, 0, 2, 1, ...] to arch ({'q': [0, 2, 4], 'k: , etc})
        x_reshape = x.reshape(self.config['n_block'], self.config['n_linear'])
        return {'self_attn.q_proj': np.array(self.q_proj_option)[x_reshape[:, 0]].tolist(),
                'self_attn.k_proj': np.array(self.k_proj_option)[x_reshape[:, 1]].tolist(),
                'self_attn.v_proj': np.array(self.v_proj_option)[x_reshape[:, 2]].tolist(),
                'self_attn.o_proj': np.array(self.o_proj_option)[x_reshape[:, 3]].tolist(),
                'mlp.down_proj': np.array(self.down_proj_option)[x_reshape[:, 4]].tolist(),
                'mlp.up_proj': np.array(self.up_proj_option)[x_reshape[:, 5]].tolist(),
                'mlp.gate_proj': np.array(self.gate_proj_option)[x_reshape[:, 6]].tolist()}
               
# def softmax(x):
#     return np.exp(x - x.max()) / np.exp(x - x.max()).sum()