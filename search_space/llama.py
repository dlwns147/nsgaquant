import numpy as np
from utils import get_net_info
from tqdm import tqdm
import math


class LlamaQuantSearchSpace:
    def __init__(self, 
                n_block,
                pass_linear_list=[],
                quant_model_bits=[],
                group_size=-1,
                outlier_bits=[],
                config=None,
                sec_obj='bits',
                sec_obj_range=[],
                only_outlier_bits=False,
                latency_table=None,
                rand_size=5):
        self.n_block = n_block  # number of blocks

        self.quant_model_bits = quant_model_bits
        
        self.q_proj_option = sorted(([b for b in quant_model_bits if b not in list(map(int, outlier_bits['self_attn.q_proj']))] if only_outlier_bits else quant_model_bits) + outlier_bits['self_attn.q_proj'])
        self.k_proj_option = sorted(([b for b in quant_model_bits if b not in list(map(int, outlier_bits['self_attn.k_proj']))] if only_outlier_bits else quant_model_bits) + outlier_bits['self_attn.k_proj'])
        self.v_proj_option = sorted(([b for b in quant_model_bits if b not in list(map(int, outlier_bits['self_attn.v_proj']))] if only_outlier_bits else quant_model_bits) + outlier_bits['self_attn.v_proj'])
        # self.v_proj_option = sorted(quant_model_bits + outlier_bits['self_attn.v_proj'])
        self.o_proj_option = quant_model_bits

        self.gate_proj_option = sorted(([b for b in quant_model_bits if b not in list(map(int, outlier_bits['mlp.gate_proj']))] if only_outlier_bits else quant_model_bits) + outlier_bits['mlp.gate_proj'])
        self.up_proj_option = sorted(([b for b in quant_model_bits if b not in list(map(int, outlier_bits['mlp.up_proj']))] if only_outlier_bits else quant_model_bits) + outlier_bits['mlp.up_proj'])
        self.down_proj_option = sorted(([b for b in quant_model_bits if b not in list(map(int, outlier_bits['mlp.down_proj']))] if only_outlier_bits else quant_model_bits) + outlier_bits['mlp.down_proj'])

        self.pass_linear_list = pass_linear_list
        self.config = config
        self.group_size = group_size
        self.latency_table = latency_table
        self.linear = config['linear']
        self.n_linear = len(self.linear)
        self.rand_size = rand_size
        
        self.sec_obj = sec_obj
        self.sec_obj_range = sec_obj_range
        assert len(sec_obj_range) == 2, f"sec_obj_range is invalid: {sec_obj_range}"
        assert math.isclose(sec_obj_range[0], sec_obj_range[1]) or sec_obj_range[0] < sec_obj_range[1], f"sec_obj_range is invalid: {sec_obj_range}"
        
        self.n_bits = len(quant_model_bits)
        self.pass_linear_idx_list = []
        for pass_linear in self.pass_linear_list:
            blk, linear = pass_linear.split('.', maxsplit=1)
            linear_idx = self.config['linear'].index(linear)
            # self.pass_linear_idx_list.append(int(blk) * self.n_linear + linear_idx)
            self.pass_linear_idx_list.append(int(blk) + self.n_block * linear_idx)
            
        self.pass_linear_idx_list.sort()
        print(f'self.pass_linear_idx_list : {self.pass_linear_idx_list}')

    def sample(self, n_samples=1, nb=None, q=None, k=None, v=None, o=None, down=None, up=None, gate=None, lp=None, pool=[]):
        """ randomly sample a architecture"""
        nb = self.n_block if nb is None else nb
        q = self.q_proj_option if q is None else q
        k = self.k_proj_option if k is None else k
        v = self.v_proj_option if v is None else v
        o = self.o_proj_option if o is None else o
        gate = self.gate_proj_option if gate is None else gate
        up = self.up_proj_option if up is None else up
        down = self.down_proj_option if down is None else down
        
        data = []
        for n in tqdm(range(n_samples), desc='Sampling'):
            while True:
                # prob = np.random.rand(3)
                prob = np.random.rand(self.rand_size)
                # max(len(q), len(k), len(v), len(o), len(gate), len(up), len(down))
                prob = np.random.rand(6)
                q_prob = prob[np.array([np.argwhere(_x == np.array(self.q_proj_option))[0, 0] for _x in q])]
                q_list = np.random.choice(q, size=nb, p=q_prob / q_prob.sum(), replace=True).tolist()

                k_prob = prob[np.array([np.argwhere(_x == np.array(self.k_proj_option))[0, 0] for _x in k])]
                k_list = np.random.choice(k, size=nb, p=k_prob / k_prob.sum(), replace=True).tolist()
                
                v_prob = prob[np.array([np.argwhere(_x == np.array(self.v_proj_option))[0, 0] for _x in v])]
                v_list = np.random.choice(v, size=nb, p=v_prob / v_prob.sum(), replace=True).tolist()
                
                o_prob = prob[np.array([np.argwhere(_x == np.array(self.o_proj_option))[0, 0] for _x in o])]
                o_list = np.random.choice(o, size=nb, p=o_prob / o_prob.sum(), replace=True).tolist()

                gate_prob = prob[np.array([np.argwhere(_x == np.array(self.gate_proj_option))[0, 0] for _x in gate])]
                gate_list = np.random.choice(gate, size=nb, p=gate_prob / gate_prob.sum(), replace=True).tolist()
                
                up_prob = prob[np.array([np.argwhere(_x == np.array(self.up_proj_option))[0, 0] for _x in up])]
                up_list = np.random.choice(up, size=nb, p=up_prob / up_prob.sum(), replace=True).tolist()

                down_prob = prob[np.array([np.argwhere(_x == np.array(self.down_proj_option))[0, 0] for _x in down])]
                down_list = np.random.choice(down, size=nb, p=down_prob / down_prob.sum(), replace=True).tolist()

                for pass_linear in self.pass_linear_list:
                    blk, linear = pass_linear.split('.')[0], pass_linear.split('.')[-1]
                    blk = int(blk)

                    if linear == 'q_proj':
                        q_list[blk] = max(self.q_proj_option)
                    elif linear == 'k_proj':
                        k_list[blk] = max(self.k_proj_option)
                    elif linear == 'v_proj':
                        v_list[blk] = max(self.v_proj_option)
                    elif linear == 'o_proj':
                        o_list[blk] = max(self.o_proj_option)

                    elif linear == 'gate_proj':
                        gate_list[blk] = max(self.gate_proj_option)
                    elif linear == 'up_proj':
                        up_list[blk] = max(self.up_proj_option)
                    elif linear == 'down_proj':
                        down_list[blk] = max(self.down_proj_option)
                    else:
                        raise NotImplementedError(f"linear : {linear}")

                    
                new_arch = {'linear': {'self_attn.q_proj': q_list, 'self_attn.k_proj': k_list, 'self_attn.v_proj': v_list, 'self_attn.o_proj': o_list, 'mlp.gate_proj': gate_list, 'mlp.up_proj': up_list, 'mlp.down_proj': down_list}}
                complexity = get_net_info(new_arch, self.config, self.group_size, self.latency_table)
                # print(f'new_arch : {new_arch}')
                # print(f'complexity : {complexity}')
                if (new_arch not in data) and \
                    (new_arch not in pool) and \
                    (math.isclose(complexity[self.sec_obj], self.sec_obj_range[0]) or complexity[self.sec_obj] > self.sec_obj_range[0]) and \
                    (math.isclose(complexity[self.sec_obj], self.sec_obj_range[1]) or complexity[self.sec_obj] < self.sec_obj_range[1]):
                    # print(f'selected arch : {complexity}')
                    # print('=' * 20)
                    break
                
            data.append(new_arch)
        return data

    def initialize(self, n_doe, pool=[]):
        # sample one arch with least (lb of hyperparameters) and most complexity (ub of hyperparameters)
        data = []
        # if math.isclose(self.sec_obj_range[0], min(self.quant_model_bits)):
        data.append(self.sample(q=[min(self.q_proj_option)], k=[min(self.k_proj_option)], v=[min(self.v_proj_option)], o=[min(self.o_proj_option)], down=[min(self.down_proj_option)], up=[min(self.up_proj_option)], gate=[min(self.gate_proj_option)])[0])
        n_doe -= 1
        # if math.isclose(self.sec_obj_range[-1], max(self.quant_model_bits)):
        data.append(self.sample(q=[max(self.q_proj_option)], k=[max(self.k_proj_option)], v=[max(self.v_proj_option)], o=[max(self.o_proj_option)], down=[max(self.down_proj_option)], up=[max(self.up_proj_option)], gate=[max(self.gate_proj_option)])[0])
        n_doe -= 1
        data.extend(self.sample(n_samples=n_doe, pool=pool))
        return data

    def encode(self, arch):
        # encode arch ({'q': [0, 2, 4], 'k: , etc}) to integer bit-string [1, 0, 2, 1, ...]
        # q_encode = np.array([np.argwhere(_x == np.array(self.q_proj_option))[0, 0] for _x in arch['linear']['self_attn.q_proj']]).reshape(-1, 1)
        # k_encode = np.array([np.argwhere(_x == np.array(self.k_proj_option))[0, 0] for _x in arch['linear']['self_attn.k_proj']]).reshape(-1, 1)
        # v_encode = np.array([np.argwhere(_x == np.array(self.v_proj_option))[0, 0] for _x in arch['linear']['self_attn.v_proj']]).reshape(-1, 1)
        # o_encode = np.array([np.argwhere(_x == np.array(self.o_proj_option))[0, 0] for _x in arch['linear']['self_attn.o_proj']]).reshape(-1, 1)
        # gate_encode = np.array([np.argwhere(_x == np.array(self.gate_proj_option))[0, 0] for _x in arch['linear']['mlp.gate_proj']]).reshape(-1, 1)
        # up_encode = np.array([np.argwhere(_x == np.array(self.up_proj_option))[0, 0] for _x in arch['linear']['mlp.up_proj']]).reshape(-1, 1)
        # down_encode = np.array([np.argwhere(_x == np.array(self.down_proj_option))[0, 0] for _x in arch['linear']['mlp.down_proj']]).reshape(-1, 1)

        # return np.stack((q_encode, k_encode, v_encode, o_encode, gate_encode, up_encode, down_encode), axis=-1).flatten()
        q_encode = np.array([np.argwhere(_x == np.array(self.q_proj_option))[0, 0] for _x in arch['linear']['self_attn.q_proj']])
        k_encode = np.array([np.argwhere(_x == np.array(self.k_proj_option))[0, 0] for _x in arch['linear']['self_attn.k_proj']])
        v_encode = np.array([np.argwhere(_x == np.array(self.v_proj_option))[0, 0] for _x in arch['linear']['self_attn.v_proj']])
        o_encode = np.array([np.argwhere(_x == np.array(self.o_proj_option))[0, 0] for _x in arch['linear']['self_attn.o_proj']])
        gate_encode = np.array([np.argwhere(_x == np.array(self.gate_proj_option))[0, 0] for _x in arch['linear']['mlp.gate_proj']])
        up_encode = np.array([np.argwhere(_x == np.array(self.up_proj_option))[0, 0] for _x in arch['linear']['mlp.up_proj']])
        down_encode = np.array([np.argwhere(_x == np.array(self.down_proj_option))[0, 0] for _x in arch['linear']['mlp.down_proj']])

        return np.concatenate((q_encode, k_encode, v_encode, o_encode, gate_encode, up_encode, down_encode))
    
    def decode(self, x):
        # decode integer bit-string [1, 0, 2, 1, ...] to arch ({'q': [0, 2, 4], 'k: , etc})
        # x_reshape = x.reshape(self.n_block, self.n_linear)
        # return {
        #             'linear': {
        #                 'self_attn.q_proj': np.array(self.q_proj_option)[x_reshape[:, 0]].tolist(),
        #                 'self_attn.k_proj': np.array(self.k_proj_option)[x_reshape[:, 1]].tolist(),
        #                 'self_attn.v_proj': np.array(self.v_proj_option)[x_reshape[:, 2]].tolist(),
        #                 'self_attn.o_proj': np.array(self.o_proj_option)[x_reshape[:, 3]].tolist(),
        #                 'mlp.gate_proj': np.array(self.gate_proj_option)[x_reshape[:, 4]].tolist(),
        #                 'mlp.up_proj': np.array(self.up_proj_option)[x_reshape[:, 5]].tolist(),
        #                 'mlp.down_proj': np.array(self.down_proj_option)[x_reshape[:, 6]].tolist(),
        #             },
        #         }
        x_reshape = x.reshape(self.n_linear, self.n_block)
        return {
                    'linear': {
                        'self_attn.q_proj': np.array(self.q_proj_option)[x_reshape[0]].tolist(),
                        'self_attn.k_proj': np.array(self.k_proj_option)[x_reshape[1]].tolist(),
                        'self_attn.v_proj': np.array(self.v_proj_option)[x_reshape[2]].tolist(),
                        'self_attn.o_proj': np.array(self.o_proj_option)[x_reshape[3]].tolist(),
                        'mlp.gate_proj': np.array(self.gate_proj_option)[x_reshape[4]].tolist(),
                        'mlp.up_proj': np.array(self.up_proj_option)[x_reshape[5]].tolist(),
                        'mlp.down_proj': np.array(self.down_proj_option)[x_reshape[6]].tolist(),
                    },
                }
    
    def encode_predictor(self, arch):
        q_encode = np.array([np.argwhere(_x == np.array(self.q_proj_option))[0, 0] for blk_idx, _x in enumerate(arch['linear']['self_attn.q_proj']) if f'{blk_idx}.self_attn.q_proj' not in self.pass_linear_list])
        k_encode = np.array([np.argwhere(_x == np.array(self.k_proj_option))[0, 0] for blk_idx, _x in enumerate(arch['linear']['self_attn.k_proj']) if f'{blk_idx}.self_attn.k_proj' not in self.pass_linear_list])
        v_encode = np.array([np.argwhere(_x == np.array(self.v_proj_option))[0, 0] for blk_idx, _x in enumerate(arch['linear']['self_attn.v_proj']) if f'{blk_idx}.self_attn.v_proj' not in self.pass_linear_list])
        o_encode = np.array([np.argwhere(_x == np.array(self.o_proj_option))[0, 0] for blk_idx, _x in enumerate(arch['linear']['self_attn.o_proj']) if f'{blk_idx}.self_attn.o_proj' not in self.pass_linear_list])
        gate_encode = np.array([np.argwhere(_x == np.array(self.gate_proj_option))[0, 0] for blk_idx, _x in enumerate(arch['linear']['mlp.gate_proj']) if f'{blk_idx}.mlp.gate_proj' not in self.pass_linear_list])
        up_encode = np.array([np.argwhere(_x == np.array(self.up_proj_option))[0, 0] for blk_idx, _x in enumerate(arch['linear']['mlp.up_proj']) if f'{blk_idx}.mlp.up_proj' not in self.pass_linear_list])
        down_encode = np.array([np.argwhere(_x == np.array(self.down_proj_option))[0, 0] for blk_idx, _x in enumerate(arch['linear']['mlp.down_proj']) if f'{blk_idx}.mlp.down_proj' not in self.pass_linear_list])

        return np.concatenate((q_encode, k_encode, v_encode, o_encode, gate_encode, up_encode, down_encode))
    
    def decode_encode_predictor(self, x): # x : (batch_size, dim)
        # B = x.shape[0]
        # x = x.reshape(B, self.n_block, self.n_linear).transpose(0, 2, 1).reshape(B, -1)
        return np.delete(x, self.pass_linear_idx_list, axis=-1)


class LlamaQEFTSearchSpace:
    def __init__(self, 
                n_block,
                pass_linear_list=[],
                quant_model_bits=[],
                group_size=-1,
                n_outlier=[],
                outlier_bits=[],
                config=None,
                sec_obj='bits',
                sec_obj_range=[],
                only_outlier_bits=False,
                latency_table=None,
                rand_size=5):
        self.n_block = n_block  # number of blocks

        self.quant_model_bits = quant_model_bits
        self.n_outlier = n_outlier
        
        self.q_proj_option = sorted(([b for b in quant_model_bits if b not in list(map(int, outlier_bits['self_attn.q_proj']))] if only_outlier_bits else quant_model_bits) + outlier_bits['self_attn.q_proj'])
        self.k_proj_option = sorted(([b for b in quant_model_bits if b not in list(map(int, outlier_bits['self_attn.k_proj']))] if only_outlier_bits else quant_model_bits) + outlier_bits['self_attn.k_proj'])
        self.v_proj_option = sorted(([b for b in quant_model_bits if b not in list(map(int, outlier_bits['self_attn.v_proj']))] if only_outlier_bits else quant_model_bits) + outlier_bits['self_attn.v_proj'])
        # self.v_proj_option = sorted(quant_model_bits + outlier_bits['self_attn.v_proj'])
        self.o_proj_option = quant_model_bits

        self.gate_proj_option = sorted(([b for b in quant_model_bits if b not in list(map(int, outlier_bits['mlp.gate_proj']))] if only_outlier_bits else quant_model_bits) + outlier_bits['mlp.gate_proj'])
        self.up_proj_option = sorted(([b for b in quant_model_bits if b not in list(map(int, outlier_bits['mlp.up_proj']))] if only_outlier_bits else quant_model_bits) + outlier_bits['mlp.up_proj'])
        self.down_proj_option = sorted(([b for b in quant_model_bits if b not in list(map(int, outlier_bits['mlp.down_proj']))] if only_outlier_bits else quant_model_bits) + outlier_bits['mlp.down_proj'])

        self.pass_linear_list = pass_linear_list
        self.config = config
        self.group_size = group_size
        self.latency_table = latency_table
        self.linear = config['linear']
        self.n_linear = len(self.linear)
        self.rand_size = rand_size
        
        self.sec_obj = sec_obj
        self.sec_obj_range = sec_obj_range
        assert len(sec_obj_range) == 2, f"sec_obj_range is invalid: {sec_obj_range}"
        assert math.isclose(sec_obj_range[0], sec_obj_range[1]) or sec_obj_range[0] < sec_obj_range[1], f"sec_obj_range is invalid: {sec_obj_range}"
        
        self.n_bits = len(quant_model_bits)
        self.pass_linear_idx_list = []
        for pass_linear in self.pass_linear_list:
            blk, linear = pass_linear.split('.', maxsplit=1)
            linear_idx = self.config['linear'].index(linear)
            # self.pass_linear_idx_list.append(int(blk) * self.n_linear + linear_idx)
            self.pass_linear_idx_list.append(int(blk) + self.n_block * linear_idx)
            
        self.pass_linear_idx_list.sort()
        print(f'self.pass_linear_idx_list : {self.pass_linear_idx_list}')

    def sample(self, n_samples=1, nb=None, q=None, k=None, v=None, o=None, down=None, up=None, gate=None, lp=None, pool=[]):
        """ randomly sample a architecture"""
        nb = self.n_block if nb is None else nb
        q = self.q_proj_option if q is None else q
        k = self.k_proj_option if k is None else k
        v = self.v_proj_option if v is None else v
        o = self.o_proj_option if o is None else o
        gate = self.gate_proj_option if gate is None else gate
        up = self.up_proj_option if up is None else up
        down = self.down_proj_option if down is None else down
        
        data = []
        for n in tqdm(range(n_samples), desc='Sampling'):
            while True:
                # prob = np.random.rand(3)
                prob = np.random.rand(self.rand_size)
                # max(len(q), len(k), len(v), len(o), len(gate), len(up), len(down))
                prob = np.random.rand(6)
                q_prob = prob[np.array([np.argwhere(_x == np.array(self.q_proj_option))[0, 0] for _x in q])]
                q_list = np.random.choice(q, size=nb, p=q_prob / q_prob.sum(), replace=True).tolist()

                k_prob = prob[np.array([np.argwhere(_x == np.array(self.k_proj_option))[0, 0] for _x in k])]
                k_list = np.random.choice(k, size=nb, p=k_prob / k_prob.sum(), replace=True).tolist()
                
                v_prob = prob[np.array([np.argwhere(_x == np.array(self.v_proj_option))[0, 0] for _x in v])]
                v_list = np.random.choice(v, size=nb, p=v_prob / v_prob.sum(), replace=True).tolist()
                
                o_prob = prob[np.array([np.argwhere(_x == np.array(self.o_proj_option))[0, 0] for _x in o])]
                o_list = np.random.choice(o, size=nb, p=o_prob / o_prob.sum(), replace=True).tolist()

                gate_prob = prob[np.array([np.argwhere(_x == np.array(self.gate_proj_option))[0, 0] for _x in gate])]
                gate_list = np.random.choice(gate, size=nb, p=gate_prob / gate_prob.sum(), replace=True).tolist()
                
                up_prob = prob[np.array([np.argwhere(_x == np.array(self.up_proj_option))[0, 0] for _x in up])]
                up_list = np.random.choice(up, size=nb, p=up_prob / up_prob.sum(), replace=True).tolist()

                down_prob = prob[np.array([np.argwhere(_x == np.array(self.down_proj_option))[0, 0] for _x in down])]
                down_list = np.random.choice(down, size=nb, p=down_prob / down_prob.sum(), replace=True).tolist()

                for pass_linear in self.pass_linear_list:
                    blk, linear = pass_linear.split('.')[0], pass_linear.split('.')[-1]
                    blk = int(blk)

                    if linear == 'q_proj':
                        q_list[blk] = max(self.q_proj_option)
                    elif linear == 'k_proj':
                        k_list[blk] = max(self.k_proj_option)
                    elif linear == 'v_proj':
                        v_list[blk] = max(self.v_proj_option)
                    elif linear == 'o_proj':
                        o_list[blk] = max(self.o_proj_option)

                    elif linear == 'gate_proj':
                        gate_list[blk] = max(self.gate_proj_option)
                    elif linear == 'up_proj':
                        up_list[blk] = max(self.up_proj_option)
                    elif linear == 'down_proj':
                        down_list[blk] = max(self.down_proj_option)
                    else:
                        raise NotImplementedError(f"linear : {linear}")

                    
                new_arch = {'linear': {'self_attn.q_proj': q_list, 'self_attn.k_proj': k_list, 'self_attn.v_proj': v_list, 'self_attn.o_proj': o_list, 'mlp.gate_proj': gate_list, 'mlp.up_proj': up_list, 'mlp.down_proj': down_list}}
                complexity = get_net_info(new_arch, self.config, self.group_size, self.latency_table)
                # print(f'new_arch : {new_arch}')
                # print(f'complexity : {complexity}')
                if (new_arch not in data) and \
                    (new_arch not in pool) and \
                    (math.isclose(complexity[self.sec_obj], self.sec_obj_range[0]) or complexity[self.sec_obj] > self.sec_obj_range[0]) and \
                    (math.isclose(complexity[self.sec_obj], self.sec_obj_range[1]) or complexity[self.sec_obj] < self.sec_obj_range[1]):
                    # print(f'selected arch : {complexity}')
                    # print('=' * 20)
                    break
                
            data.append(new_arch)
        return data

    def initialize(self, n_doe, pool=[]):
        # sample one arch with least (lb of hyperparameters) and most complexity (ub of hyperparameters)
        data = []
        # if math.isclose(self.sec_obj_range[0], min(self.quant_model_bits)):
        data.append(self.sample(q=[min(self.q_proj_option)], k=[min(self.k_proj_option)], v=[min(self.v_proj_option)], o=[min(self.o_proj_option)], down=[min(self.down_proj_option)], up=[min(self.up_proj_option)], gate=[min(self.gate_proj_option)])[0])
        n_doe -= 1
        # if math.isclose(self.sec_obj_range[-1], max(self.quant_model_bits)):
        data.append(self.sample(q=[max(self.q_proj_option)], k=[max(self.k_proj_option)], v=[max(self.v_proj_option)], o=[max(self.o_proj_option)], down=[max(self.down_proj_option)], up=[max(self.up_proj_option)], gate=[max(self.gate_proj_option)])[0])
        n_doe -= 1
        data.extend(self.sample(n_samples=n_doe, pool=pool))
        return data

    def encode(self, arch):
        # encode arch ({'q': [0, 2, 4], 'k: , etc}) to integer bit-string [1, 0, 2, 1, ...]
        # q_encode = np.array([np.argwhere(_x == np.array(self.q_proj_option))[0, 0] for _x in arch['linear']['self_attn.q_proj']]).reshape(-1, 1)
        # k_encode = np.array([np.argwhere(_x == np.array(self.k_proj_option))[0, 0] for _x in arch['linear']['self_attn.k_proj']]).reshape(-1, 1)
        # v_encode = np.array([np.argwhere(_x == np.array(self.v_proj_option))[0, 0] for _x in arch['linear']['self_attn.v_proj']]).reshape(-1, 1)
        # o_encode = np.array([np.argwhere(_x == np.array(self.o_proj_option))[0, 0] for _x in arch['linear']['self_attn.o_proj']]).reshape(-1, 1)
        # gate_encode = np.array([np.argwhere(_x == np.array(self.gate_proj_option))[0, 0] for _x in arch['linear']['mlp.gate_proj']]).reshape(-1, 1)
        # up_encode = np.array([np.argwhere(_x == np.array(self.up_proj_option))[0, 0] for _x in arch['linear']['mlp.up_proj']]).reshape(-1, 1)
        # down_encode = np.array([np.argwhere(_x == np.array(self.down_proj_option))[0, 0] for _x in arch['linear']['mlp.down_proj']]).reshape(-1, 1)

        # return np.stack((q_encode, k_encode, v_encode, o_encode, gate_encode, up_encode, down_encode), axis=-1).flatten()
        q_encode = np.array([np.argwhere(_x == np.array(self.q_proj_option))[0, 0] for _x in arch['linear']['self_attn.q_proj']])
        k_encode = np.array([np.argwhere(_x == np.array(self.k_proj_option))[0, 0] for _x in arch['linear']['self_attn.k_proj']])
        v_encode = np.array([np.argwhere(_x == np.array(self.v_proj_option))[0, 0] for _x in arch['linear']['self_attn.v_proj']])
        o_encode = np.array([np.argwhere(_x == np.array(self.o_proj_option))[0, 0] for _x in arch['linear']['self_attn.o_proj']])
        gate_encode = np.array([np.argwhere(_x == np.array(self.gate_proj_option))[0, 0] for _x in arch['linear']['mlp.gate_proj']])
        up_encode = np.array([np.argwhere(_x == np.array(self.up_proj_option))[0, 0] for _x in arch['linear']['mlp.up_proj']])
        down_encode = np.array([np.argwhere(_x == np.array(self.down_proj_option))[0, 0] for _x in arch['linear']['mlp.down_proj']])

        return np.concatenate((q_encode, k_encode, v_encode, o_encode, gate_encode, up_encode, down_encode))
    
    def decode(self, x):
        # decode integer bit-string [1, 0, 2, 1, ...] to arch ({'q': [0, 2, 4], 'k: , etc})
        # x_reshape = x.reshape(self.n_block, self.n_linear)
        # return {
        #             'linear': {
        #                 'self_attn.q_proj': np.array(self.q_proj_option)[x_reshape[:, 0]].tolist(),
        #                 'self_attn.k_proj': np.array(self.k_proj_option)[x_reshape[:, 1]].tolist(),
        #                 'self_attn.v_proj': np.array(self.v_proj_option)[x_reshape[:, 2]].tolist(),
        #                 'self_attn.o_proj': np.array(self.o_proj_option)[x_reshape[:, 3]].tolist(),
        #                 'mlp.gate_proj': np.array(self.gate_proj_option)[x_reshape[:, 4]].tolist(),
        #                 'mlp.up_proj': np.array(self.up_proj_option)[x_reshape[:, 5]].tolist(),
        #                 'mlp.down_proj': np.array(self.down_proj_option)[x_reshape[:, 6]].tolist(),
        #             },
        #         }
        x_reshape = x.reshape(self.n_linear, self.n_block)
        return {
                    'linear': {
                        'self_attn.q_proj': np.array(self.q_proj_option)[x_reshape[0]].tolist(),
                        'self_attn.k_proj': np.array(self.k_proj_option)[x_reshape[1]].tolist(),
                        'self_attn.v_proj': np.array(self.v_proj_option)[x_reshape[2]].tolist(),
                        'self_attn.o_proj': np.array(self.o_proj_option)[x_reshape[3]].tolist(),
                        'mlp.gate_proj': np.array(self.gate_proj_option)[x_reshape[4]].tolist(),
                        'mlp.up_proj': np.array(self.up_proj_option)[x_reshape[5]].tolist(),
                        'mlp.down_proj': np.array(self.down_proj_option)[x_reshape[6]].tolist(),
                    },
                }
    
    def encode_predictor(self, arch):
        q_encode = np.array([np.argwhere(_x == np.array(self.q_proj_option))[0, 0] for blk_idx, _x in enumerate(arch['linear']['self_attn.q_proj']) if f'{blk_idx}.self_attn.q_proj' not in self.pass_linear_list])
        k_encode = np.array([np.argwhere(_x == np.array(self.k_proj_option))[0, 0] for blk_idx, _x in enumerate(arch['linear']['self_attn.k_proj']) if f'{blk_idx}.self_attn.k_proj' not in self.pass_linear_list])
        v_encode = np.array([np.argwhere(_x == np.array(self.v_proj_option))[0, 0] for blk_idx, _x in enumerate(arch['linear']['self_attn.v_proj']) if f'{blk_idx}.self_attn.v_proj' not in self.pass_linear_list])
        o_encode = np.array([np.argwhere(_x == np.array(self.o_proj_option))[0, 0] for blk_idx, _x in enumerate(arch['linear']['self_attn.o_proj']) if f'{blk_idx}.self_attn.o_proj' not in self.pass_linear_list])
        gate_encode = np.array([np.argwhere(_x == np.array(self.gate_proj_option))[0, 0] for blk_idx, _x in enumerate(arch['linear']['mlp.gate_proj']) if f'{blk_idx}.mlp.gate_proj' not in self.pass_linear_list])
        up_encode = np.array([np.argwhere(_x == np.array(self.up_proj_option))[0, 0] for blk_idx, _x in enumerate(arch['linear']['mlp.up_proj']) if f'{blk_idx}.mlp.up_proj' not in self.pass_linear_list])
        down_encode = np.array([np.argwhere(_x == np.array(self.down_proj_option))[0, 0] for blk_idx, _x in enumerate(arch['linear']['mlp.down_proj']) if f'{blk_idx}.mlp.down_proj' not in self.pass_linear_list])

        return np.concatenate((q_encode, k_encode, v_encode, o_encode, gate_encode, up_encode, down_encode))
    
    def decode_encode_predictor(self, x): # x : (batch_size, dim)
        # B = x.shape[0]
        # x = x.reshape(B, self.n_block, self.n_linear).transpose(0, 2, 1).reshape(B, -1)
        return np.delete(x, self.pass_linear_idx_list, axis=-1)