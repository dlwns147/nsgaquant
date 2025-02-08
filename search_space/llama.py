import numpy as np
from utils import get_net_info
from tqdm import tqdm
import math


class LlamaSearchSpace:
    def __init__(self, 
                n_block,
                pass_linear_list=[],
                pass_layer_list=[],
                quant_model_bits=[],
                outlier_bits=[],
                config=None,
                sec_obj='bits',
                sec_obj_range=[],
                layer_prune_range=[],
                only_outlier_bits=False,
                latency_table=None):
        self.n_block = n_block  # number of blocks

        self.quant_model_bits = quant_model_bits

        # [bits for bits in quant_model_bits if list(map(int, outlier_bits['self_attn.q_proj']))]
        
        self.q_proj_option = sorted(([b for b in quant_model_bits if b not in list(map(int, outlier_bits['self_attn.q_proj']))] if only_outlier_bits else quant_model_bits) + outlier_bits['self_attn.q_proj'])
        self.k_proj_option = sorted(([b for b in quant_model_bits if b not in list(map(int, outlier_bits['self_attn.k_proj']))] if only_outlier_bits else quant_model_bits) + outlier_bits['self_attn.k_proj'])
        self.v_proj_option = sorted(([b for b in quant_model_bits if b not in list(map(int, outlier_bits['self_attn.v_proj']))] if only_outlier_bits else quant_model_bits) + outlier_bits['self_attn.v_proj'])
        # self.v_proj_option = sorted(quant_model_bits + outlier_bits['self_attn.v_proj'])
        self.o_proj_option = quant_model_bits

        self.gate_proj_option = sorted(([b for b in quant_model_bits if b not in list(map(int, outlier_bits['mlp.gate_proj']))] if only_outlier_bits else quant_model_bits) + outlier_bits['mlp.gate_proj'])
        self.up_proj_option = sorted(([b for b in quant_model_bits if b not in list(map(int, outlier_bits['mlp.up_proj']))] if only_outlier_bits else quant_model_bits) + outlier_bits['mlp.up_proj'])
        self.down_proj_option = sorted(([b for b in quant_model_bits if b not in list(map(int, outlier_bits['mlp.down_proj']))] if only_outlier_bits else quant_model_bits) + outlier_bits['mlp.down_proj'])

        self.q_proj_option_nonzero = [b for b in self.q_proj_option if b != 0]
        self.k_proj_option_nonzero = [b for b in self.k_proj_option if b != 0]
        self.v_proj_option_nonzero = [b for b in self.v_proj_option if b != 0]
        self.o_proj_option_nonzero = [b for b in self.o_proj_option if b != 0]
        self.gate_proj_option_nonzero = [b for b in self.gate_proj_option if b != 0]
        self.up_proj_option_nonzero = [b for b in self.up_proj_option if b != 0]
        self.down_proj_option_nonzero = [b for b in self.down_proj_option if b != 0]

        self.layer_option = [0, 1]
        self.pass_linear_list = pass_linear_list
        self.pass_layer_list = pass_layer_list
        self.config = config
        self.latency_table = latency_table
        self.linear_group = config['linear']
        self.n_linear = len(self.linear_group)
        self.n_layer = int(config['n_layer'])
        
        self.sec_obj = sec_obj
        self.sec_obj_range = sec_obj_range
        assert len(sec_obj_range) == 2, f"sec_obj_range is invalid: {sec_obj_range}"
        assert math.isclose(sec_obj_range[0], sec_obj_range[1]) or sec_obj_range[0] < sec_obj_range[1], f"sec_obj_range is invalid: {sec_obj_range}"
        
        self.layer_prune_range = layer_prune_range
        assert len(layer_prune_range) == 2, f"layer_prune_range is invalid: {sec_obj_range}"
        assert math.isclose(layer_prune_range[0], layer_prune_range[1]) or layer_prune_range[0] < layer_prune_range[1], f"layer_prune_range is invalid: {layer_prune_range}"

        self.n_bits = len(quant_model_bits)

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
        lp = self.layer_prune_range if lp is None else lp
        
        data = []
        for n in tqdm(range(n_samples), desc='Sampling'):
            while True:
                # prob = np.random.rand(3)
                prob = np.random.rand(5)
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

                # gate_prob = np.random.rand(len(gate))
                gate_prob = prob[np.array([np.argwhere(_x == np.array(self.gate_proj_option))[0, 0] for _x in gate])]
                gate_list = np.random.choice(gate, size=nb, p=gate_prob / gate_prob.sum(), replace=True).tolist()
                
                # up_prob = np.random.rand(len(up))
                up_prob = prob[np.array([np.argwhere(_x == np.array(self.up_proj_option))[0, 0] for _x in up])]
                up_list = np.random.choice(up, size=nb, p=up_prob / up_prob.sum(), replace=True).tolist()

                # down_prob = np.random.rand(len(down))
                down_prob = prob[np.array([np.argwhere(_x == np.array(self.down_proj_option))[0, 0] for _x in down])]
                down_list = np.random.choice(down, size=nb, p=down_prob / down_prob.sum(), replace=True).tolist()

                remain_prob = np.random.uniform(lp[0], lp[1])
                attn_layer_list = np.random.binomial(1, remain_prob, self.n_block).tolist()
                mlp_layer_list = np.random.binomial(1, remain_prob, self.n_block).tolist()

                for pass_linear in self.pass_linear_list:
                    blk, linear = pass_linear.split('.')[0], pass_linear.split('.')[-1]
                    blk = int(blk)

                    if linear == 'q_proj':
                        q_list[blk] = max(self.q_proj_option)
                        attn_layer_list[blk] = 1
                    elif linear == 'k_proj':
                        k_list[blk] = max(self.k_proj_option)
                        attn_layer_list[blk] = 1
                    elif linear == 'v_proj':
                        v_list[blk] = max(self.v_proj_option)
                        attn_layer_list[blk] = 1
                    elif linear == 'o_proj':
                        o_list[blk] = max(self.o_proj_option)
                        attn_layer_list[blk] = 1

                    elif linear == 'gate_proj':
                        gate_list[blk] = max(self.gate_proj_option)
                        mlp_layer_list[blk] = 1
                    elif linear == 'up_proj':
                        up_list[blk] = max(self.up_proj_option)
                        mlp_layer_list[blk] = 1
                    elif linear == 'down_proj':
                        down_list[blk] = max(self.down_proj_option)
                        mlp_layer_list[blk] = 1
                    else:
                        raise NotImplementedError(f"linear : {linear}")

                for pass_layer in self.pass_layer_list:
                    blk, layer = pass_layer.split('.')
                    blk = int(blk)

                    if layer == 'self_attn':
                        attn_layer_list[blk] = 1
                    elif layer == 'mlp':
                        mlp_layer_list[blk] = 1
                    else:
                        raise NotImplementedError(f"layer : {layer}")
                    
                new_arch = {'linear': {'self_attn.q_proj': q_list, 'self_attn.k_proj': k_list, 'self_attn.v_proj': v_list, 'self_attn.o_proj': o_list, 'mlp.gate_proj': gate_list, 'mlp.up_proj': up_list, 'mlp.down_proj': down_list}, 'layer': {'self_attn': attn_layer_list, 'mlp': mlp_layer_list}}
                complexity = get_net_info(new_arch, self.config, self.latency_table)
                # print(f'new_arch : {new_arch}')
                # print(f'complexity : {complexity}')
                if (new_arch not in data) and \
                    (new_arch not in pool) and \
                    (math.isclose(complexity[self.sec_obj], self.sec_obj_range[0]) or complexity[self.sec_obj] > self.sec_obj_range[0]) and \
                    (math.isclose(complexity[self.sec_obj], self.sec_obj_range[1]) or complexity[self.sec_obj] < self.sec_obj_range[1]) and \
                    (math.isclose(complexity['sparsity'], self.layer_prune_range[0]) or complexity['sparsity'] > self.layer_prune_range[0]) and \
                    (math.isclose(complexity['sparsity'], self.layer_prune_range[1]) or complexity['sparsity'] < self.layer_prune_range[1]) :
                    # print(f'selected arch : {complexity}')
                    # print('=' * 20)
                    break
                
            data.append(new_arch)
        return data

    def initialize(self, n_doe, pool=[]):
        # sample one arch with least (lb of hyperparameters) and most complexity (ub of hyperparameters)
        data = []
        # if math.isclose(self.sec_obj_range[0], min(self.quant_model_bits)):
        # data.append(self.sample(q=[min(self.q_proj_option)], k=[min(self.k_proj_option)], v=[min(self.v_proj_option)], o=[min(self.o_proj_option)], down=[min(self.down_proj_option)], up=[min(self.up_proj_option)], gate=[min(self.gate_proj_option)], lp=[self.layer_prune_range[1], self.layer_prune_range[1]])[0])
        data.append(self.sample(q=[min(self.q_proj_option_nonzero)], k=[min(self.k_proj_option_nonzero)], v=[min(self.v_proj_option_nonzero)], o=[min(self.o_proj_option_nonzero)], down=[min(self.down_proj_option_nonzero)], up=[min(self.up_proj_option_nonzero)], gate=[min(self.gate_proj_option_nonzero)], lp=[self.layer_prune_range[1], self.layer_prune_range[1]])[0])
        n_doe -= 1
        # if math.isclose(self.sec_obj_range[-1], max(self.quant_model_bits)):
        data.append(self.sample(q=[max(self.q_proj_option)], k=[max(self.k_proj_option)], v=[max(self.v_proj_option)], o=[max(self.o_proj_option)], down=[max(self.down_proj_option)], up=[max(self.up_proj_option)], gate=[max(self.gate_proj_option)], lp=[self.layer_prune_range[1], self.layer_prune_range[1]])[0])
        n_doe -= 1
        data.extend(self.sample(q=self.q_proj_option_nonzero, k=self.k_proj_option_nonzero, v=self.v_proj_option_nonzero, o=self.o_proj_option_nonzero, gate=self.gate_proj_option_nonzero, up=self.up_proj_option_nonzero, down=self.down_proj_option_nonzero, n_samples=n_doe, pool=pool))
        return data

    def encode(self, arch):
        # encode arch ({'q': [0, 2, 4], 'k: , etc}) to integer bit-string [1, 0, 2, 1, ...]
        q_encode = np.array([np.argwhere(_x == np.array(self.q_proj_option))[0, 0] for _x in arch['linear']['self_attn.q_proj']]).reshape(-1, 1)
        k_encode = np.array([np.argwhere(_x == np.array(self.k_proj_option))[0, 0] for _x in arch['linear']['self_attn.k_proj']]).reshape(-1, 1)
        v_encode = np.array([np.argwhere(_x == np.array(self.v_proj_option))[0, 0] for _x in arch['linear']['self_attn.v_proj']]).reshape(-1, 1)
        o_encode = np.array([np.argwhere(_x == np.array(self.o_proj_option))[0, 0] for _x in arch['linear']['self_attn.o_proj']]).reshape(-1, 1)
        gate_encode = np.array([np.argwhere(_x == np.array(self.gate_proj_option))[0, 0] for _x in arch['linear']['mlp.gate_proj']]).reshape(-1, 1)
        up_encode = np.array([np.argwhere(_x == np.array(self.up_proj_option))[0, 0] for _x in arch['linear']['mlp.up_proj']]).reshape(-1, 1)
        down_encode = np.array([np.argwhere(_x == np.array(self.down_proj_option))[0, 0] for _x in arch['linear']['mlp.down_proj']]).reshape(-1, 1)
        attn_encode = np.array([np.argwhere(_x == np.array(self.layer_option))[0, 0] for _x in arch['layer']['self_attn']]).reshape(-1, 1)
        mlp_encode = np.array([np.argwhere(_x == np.array(self.layer_option))[0, 0] for _x in arch['layer']['mlp']]).reshape(-1, 1)

        return np.stack((q_encode, k_encode, v_encode, o_encode, gate_encode, up_encode, down_encode, attn_encode, mlp_encode), axis=-1).flatten()
    
    def encode_predictor(self, arch):
        
        attn_encode = np.array([np.argwhere(_x == np.array(self.layer_option))[0, 0] for _x in arch['layer']['self_attn']]).reshape(-1, 1)
        mlp_encode = np.array([np.argwhere(_x == np.array(self.layer_option))[0, 0] for _x in arch['layer']['mlp']]).reshape(-1, 1)
        
        q_encode = np.array([np.argwhere(_x == np.array(self.q_proj_option))[0, 0] for _x in arch['linear']['self_attn.q_proj']]).reshape(-1, 1) * attn_encode
        k_encode = np.array([np.argwhere(_x == np.array(self.k_proj_option))[0, 0] for _x in arch['linear']['self_attn.k_proj']]).reshape(-1, 1) * attn_encode
        v_encode = np.array([np.argwhere(_x == np.array(self.v_proj_option))[0, 0] for _x in arch['linear']['self_attn.v_proj']]).reshape(-1, 1) * attn_encode
        o_encode = np.array([np.argwhere(_x == np.array(self.o_proj_option))[0, 0] for _x in arch['linear']['self_attn.o_proj']]).reshape(-1, 1) * attn_encode
        gate_encode = np.array([np.argwhere(_x == np.array(self.gate_proj_option))[0, 0] for _x in arch['linear']['mlp.gate_proj']]).reshape(-1, 1) * mlp_encode
        up_encode = np.array([np.argwhere(_x == np.array(self.up_proj_option))[0, 0] for _x in arch['linear']['mlp.up_proj']]).reshape(-1, 1) * mlp_encode
        down_encode = np.array([np.argwhere(_x == np.array(self.down_proj_option))[0, 0] for _x in arch['linear']['mlp.down_proj']]).reshape(-1, 1) * mlp_encode

        return np.stack((q_encode, k_encode, v_encode, o_encode, gate_encode, up_encode, down_encode), axis=-1).flatten()
    
    def decode_encode_predictor(self, x):
        batch = x.shape[0]
        x = x.reshape(batch, self.n_block, self.n_linear + self.n_layer)
        x, x_layer = x[:, :, :self.n_linear], x[:, :, self.n_linear:]
        # x[:, :, :4] *= np.exapdn_dims(x_layer[:, :, 0], -1)
        # x[:, :, 4:] *= np.exapdn_dims(x_layer[:, :, 1], -1)
        x[:, :, :4] *= x_layer[:, :, 0][..., None]
        x[:, :, 4:] *= x_layer[:, :, 1][..., None]
        return x.reshape(batch, -1)

    def decode(self, x):
        # decode integer bit-string [1, 0, 2, 1, ...] to arch ({'q': [0, 2, 4], 'k: , etc})
        x_reshape = x.reshape(self.n_block, self.n_linear + self.n_layer)
        return {
                    'linear': {
                        'self_attn.q_proj': np.array(self.q_proj_option)[x_reshape[:, 0]].tolist(),
                        'self_attn.k_proj': np.array(self.k_proj_option)[x_reshape[:, 1]].tolist(),
                        'self_attn.v_proj': np.array(self.v_proj_option)[x_reshape[:, 2]].tolist(),
                        'self_attn.o_proj': np.array(self.o_proj_option)[x_reshape[:, 3]].tolist(),
                        'mlp.gate_proj': np.array(self.gate_proj_option)[x_reshape[:, 4]].tolist(),
                        'mlp.up_proj': np.array(self.up_proj_option)[x_reshape[:, 5]].tolist(),
                        'mlp.down_proj': np.array(self.down_proj_option)[x_reshape[:, 6]].tolist(),
                    },
                    'layer': {
                        'self_attn': np.array(self.layer_option)[x_reshape[:, 7]].tolist(),
                        'mlp': np.array(self.layer_option)[x_reshape[:, 8]].tolist()
                    }
                }

class LlamaQuantSearchSpace:
    def __init__(self, 
                n_block,
                pass_linear_list=[],
                quant_model_bits=[],
                outlier_bits=[],
                config=None,
                sec_obj='bits',
                sec_obj_range=[],
                only_outlier_bits=False,
                latency_table=None):
        self.n_block = n_block  # number of blocks

        self.quant_model_bits = quant_model_bits

        [bits for bits in quant_model_bits if list(map(int, outlier_bits['self_attn.q_proj']))]
        
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
        self.latency_table = latency_table
        self.linear_group = config['linear']
        self.n_linear = len(self.linear_group)
        
        self.sec_obj = sec_obj
        self.sec_obj_range = sec_obj_range
        assert len(sec_obj_range) == 2, f"sec_obj_range is invalid: {sec_obj_range}"
        assert math.isclose(sec_obj_range[0], sec_obj_range[1]) or sec_obj_range[0] < sec_obj_range[1], f"sec_obj_range is invalid: {sec_obj_range}"
        
        self.n_bits = len(quant_model_bits)
        self.pass_linear_idx_list = []
        for pass_linear in self.pass_linear_list:
            blk, linear = pass_linear.split('.', maxsplit=1)
            linear_idx = self.config['linear'].index(linear)
            self.pass_linear_idx_list.append(int(blk) * self.n_linear + linear_idx)
            
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
                prob = np.random.rand(5)
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

                # gate_prob = np.random.rand(len(gate))
                gate_prob = prob[np.array([np.argwhere(_x == np.array(self.gate_proj_option))[0, 0] for _x in gate])]
                gate_list = np.random.choice(gate, size=nb, p=gate_prob / gate_prob.sum(), replace=True).tolist()
                
                # up_prob = np.random.rand(len(up))
                up_prob = prob[np.array([np.argwhere(_x == np.array(self.up_proj_option))[0, 0] for _x in up])]
                up_list = np.random.choice(up, size=nb, p=up_prob / up_prob.sum(), replace=True).tolist()

                # down_prob = np.random.rand(len(down))
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
                complexity = get_net_info(new_arch, self.config, self.latency_table)
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
        q_encode = np.array([np.argwhere(_x == np.array(self.q_proj_option))[0, 0] for _x in arch['linear']['self_attn.q_proj']]).reshape(-1, 1)
        k_encode = np.array([np.argwhere(_x == np.array(self.k_proj_option))[0, 0] for _x in arch['linear']['self_attn.k_proj']]).reshape(-1, 1)
        v_encode = np.array([np.argwhere(_x == np.array(self.v_proj_option))[0, 0] for _x in arch['linear']['self_attn.v_proj']]).reshape(-1, 1)
        o_encode = np.array([np.argwhere(_x == np.array(self.o_proj_option))[0, 0] for _x in arch['linear']['self_attn.o_proj']]).reshape(-1, 1)
        gate_encode = np.array([np.argwhere(_x == np.array(self.gate_proj_option))[0, 0] for _x in arch['linear']['mlp.gate_proj']]).reshape(-1, 1)
        up_encode = np.array([np.argwhere(_x == np.array(self.up_proj_option))[0, 0] for _x in arch['linear']['mlp.up_proj']]).reshape(-1, 1)
        down_encode = np.array([np.argwhere(_x == np.array(self.down_proj_option))[0, 0] for _x in arch['linear']['mlp.down_proj']]).reshape(-1, 1)

        return np.stack((q_encode, k_encode, v_encode, o_encode, gate_encode, up_encode, down_encode), axis=-1).flatten()
    
    def decode(self, x):
        # decode integer bit-string [1, 0, 2, 1, ...] to arch ({'q': [0, 2, 4], 'k: , etc})
        x_reshape = x.reshape(self.n_block, self.n_linear)
        return {
                    'linear': {
                        'self_attn.q_proj': np.array(self.q_proj_option)[x_reshape[:, 0]].tolist(),
                        'self_attn.k_proj': np.array(self.k_proj_option)[x_reshape[:, 1]].tolist(),
                        'self_attn.v_proj': np.array(self.v_proj_option)[x_reshape[:, 2]].tolist(),
                        'self_attn.o_proj': np.array(self.o_proj_option)[x_reshape[:, 3]].tolist(),
                        'mlp.gate_proj': np.array(self.gate_proj_option)[x_reshape[:, 4]].tolist(),
                        'mlp.up_proj': np.array(self.up_proj_option)[x_reshape[:, 5]].tolist(),
                        'mlp.down_proj': np.array(self.down_proj_option)[x_reshape[:, 6]].tolist(),
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
        x = np.delete(x, self.pass_linear_idx_list, axis=-1)
        return x


class LlamaQuantMultiObjSearchSpace:
    def __init__(self, 
                n_block,
                pass_linear_list=[],
                quant_model_bits=[],
                outlier_bits=[],
                config=None,
                comp_obj='bits',
                comp_obj_min=[],
                comp_obj_max=[],
                only_outlier_bits=False,
                latency_table=None):
        self.n_block = n_block  # number of blocks

        self.quant_model_bits = quant_model_bits

        [bits for bits in quant_model_bits if list(map(int, outlier_bits['self_attn.q_proj']))]
        
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
        self.latency_table = latency_table
        self.n_linear = len(config['linear'])
        self.n_layer = int(config['n_layer'])
        
        self.comp_obj = comp_obj
        self.comp_obj_min = comp_obj_min
        self.comp_obj_max = comp_obj_max
        
        self.n_bits = len(quant_model_bits)

    def sample(self, n_samples=1, nb=None, q=None, k=None, v=None, o=None, down=None, up=None, gate=None, pool=[]):
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
                prob = np.random.rand(5)
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

                # gate_prob = np.random.rand(len(gate))
                gate_prob = prob[np.array([np.argwhere(_x == np.array(self.gate_proj_option))[0, 0] for _x in gate])]
                gate_list = np.random.choice(gate, size=nb, p=gate_prob / gate_prob.sum(), replace=True).tolist()
                
                # up_prob = np.random.rand(len(up))
                up_prob = prob[np.array([np.argwhere(_x == np.array(self.up_proj_option))[0, 0] for _x in up])]
                up_list = np.random.choice(up, size=nb, p=up_prob / up_prob.sum(), replace=True).tolist()

                # down_prob = np.random.rand(len(down))
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
                complexity = get_net_info(new_arch, self.config, self.latency_table)
                # print(f'new_arch : {new_arch}')
                # print(f'complexity : {complexity}')
                flag = (new_arch not in data) and (new_arch not in pool)
                for i, obj in enumerate(self.comp_obj):
                    flag &= (math.isclose(complexity[obj], self.comp_obj_min[i]) or complexity[obj] > self.comp_obj_min[i]) and \
                            (math.isclose(complexity[obj], self.comp_obj_max[i]) or complexity[obj] < self.comp_obj_max[i])
                if flag:
                    break

                # if (new_arch not in data) and \
                #     (new_arch not in pool) and \
                #     (math.isclose(complexity[self.sec_obj], self.sec_obj_range[0]) or complexity[self.sec_obj] > self.sec_obj_range[0]) and \
                #     (math.isclose(complexity[self.sec_obj], self.sec_obj_range[1]) or complexity[self.sec_obj] < self.sec_obj_range[1]) and \
                #     (math.isclose(complexity['sparsity'], self.layer_prune_range[0]) or complexity['sparsity'] > self.layer_prune_range[0]) and \
                #     (math.isclose(complexity['sparsity'], self.layer_prune_range[1]) or complexity['sparsity'] < self.layer_prune_range[1]) :
                #     # print(f'selected arch : {complexity}')
                #     # print('=' * 20)
                #     break
                
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
        q_encode = np.array([np.argwhere(_x == np.array(self.q_proj_option))[0, 0] for _x in arch['linear']['self_attn.q_proj']]).reshape(-1, 1)
        k_encode = np.array([np.argwhere(_x == np.array(self.k_proj_option))[0, 0] for _x in arch['linear']['self_attn.k_proj']]).reshape(-1, 1)
        v_encode = np.array([np.argwhere(_x == np.array(self.v_proj_option))[0, 0] for _x in arch['linear']['self_attn.v_proj']]).reshape(-1, 1)
        o_encode = np.array([np.argwhere(_x == np.array(self.o_proj_option))[0, 0] for _x in arch['linear']['self_attn.o_proj']]).reshape(-1, 1)
        gate_encode = np.array([np.argwhere(_x == np.array(self.gate_proj_option))[0, 0] for _x in arch['linear']['mlp.gate_proj']]).reshape(-1, 1)
        up_encode = np.array([np.argwhere(_x == np.array(self.up_proj_option))[0, 0] for _x in arch['linear']['mlp.up_proj']]).reshape(-1, 1)
        down_encode = np.array([np.argwhere(_x == np.array(self.down_proj_option))[0, 0] for _x in arch['linear']['mlp.down_proj']]).reshape(-1, 1)

        return np.stack((q_encode, k_encode, v_encode, o_encode, gate_encode, up_encode, down_encode), axis=-1).flatten()
    
    # def encode_predictor(self, arch):
        
    #     attn_encode = np.array([np.argwhere(_x == np.array(self.layer_option))[0, 0] for _x in arch['layer']['self_attn']]).reshape(-1, 1)
    #     mlp_encode = np.array([np.argwhere(_x == np.array(self.layer_option))[0, 0] for _x in arch['layer']['mlp']]).reshape(-1, 1)
        
    #     q_encode = np.array([np.argwhere(_x == np.array(self.q_proj_option))[0, 0] for _x in arch['linear']['self_attn.q_proj']]).reshape(-1, 1) * attn_encode
    #     k_encode = np.array([np.argwhere(_x == np.array(self.k_proj_option))[0, 0] for _x in arch['linear']['self_attn.k_proj']]).reshape(-1, 1) * attn_encode
    #     v_encode = np.array([np.argwhere(_x == np.array(self.v_proj_option))[0, 0] for _x in arch['linear']['self_attn.v_proj']]).reshape(-1, 1) * attn_encode
    #     o_encode = np.array([np.argwhere(_x == np.array(self.o_proj_option))[0, 0] for _x in arch['linear']['self_attn.o_proj']]).reshape(-1, 1) * attn_encode
    #     gate_encode = np.array([np.argwhere(_x == np.array(self.gate_proj_option))[0, 0] for _x in arch['linear']['mlp.gate_proj']]).reshape(-1, 1) * mlp_encode
    #     up_encode = np.array([np.argwhere(_x == np.array(self.up_proj_option))[0, 0] for _x in arch['linear']['mlp.up_proj']]).reshape(-1, 1) * mlp_encode
    #     down_encode = np.array([np.argwhere(_x == np.array(self.down_proj_option))[0, 0] for _x in arch['linear']['mlp.down_proj']]).reshape(-1, 1) * mlp_encode

    #     return np.stack((q_encode, k_encode, v_encode, o_encode, gate_encode, up_encode, down_encode), axis=-1).flatten()

    def decode(self, x):
        # decode integer bit-string [1, 0, 2, 1, ...] to arch ({'q': [0, 2, 4], 'k: , etc})
        x_reshape = x.reshape(self.n_block, self.n_linear)
        return {
                    'linear': {
                        'self_attn.q_proj': np.array(self.q_proj_option)[x_reshape[:, 0]].tolist(),
                        'self_attn.k_proj': np.array(self.k_proj_option)[x_reshape[:, 1]].tolist(),
                        'self_attn.v_proj': np.array(self.v_proj_option)[x_reshape[:, 2]].tolist(),
                        'self_attn.o_proj': np.array(self.o_proj_option)[x_reshape[:, 3]].tolist(),
                        'mlp.gate_proj': np.array(self.gate_proj_option)[x_reshape[:, 4]].tolist(),
                        'mlp.up_proj': np.array(self.up_proj_option)[x_reshape[:, 5]].tolist(),
                        'mlp.down_proj': np.array(self.down_proj_option)[x_reshape[:, 6]].tolist(),
                    },
                }


class LlamaLayerSearchSpace:
    def __init__(self, 
                n_block,
                pass_layer_list=[],
                config=None,
                sec_obj='sparsity',
                sec_obj_range=[],
                layer_prune_range=[],
                latency_table=None):
        self.n_block = n_block  # number of blocks

        self.layer_option = [0, 1]
        self.pass_layer_list = pass_layer_list
        self.config = config
        self.latency_table = latency_table
        self.n_layer = int(config['n_layer'])
        
        self.sec_obj = sec_obj
        self.sec_obj_range = sec_obj_range
        assert len(sec_obj_range) == 2, f"sec_obj_range is invalid: {sec_obj_range}"
        assert math.isclose(sec_obj_range[0], sec_obj_range[1]) or sec_obj_range[0] < sec_obj_range[1], f"sec_obj_range is invalid: {sec_obj_range}"
        
        self.layer_prune_range = layer_prune_range
        assert len(layer_prune_range) == 2, f"layer_prune_range is invalid: {sec_obj_range}"
        assert math.isclose(layer_prune_range[0], layer_prune_range[1]) or layer_prune_range[0] < layer_prune_range[1], f"layer_prune_range is invalid: {layer_prune_range}"

    def sample(self, n_samples=1, nb=None, lp=None, pool=[]):
        """ randomly sample a architecture"""
        nb = self.n_block if nb is None else nb
        lp = self.layer_prune_range if lp is None else lp
        
        data = []
        # for n in tqdm(range(n_samples), desc='Sampling'):
        for n in range(n_samples):
            while True:

                remain_prob = np.random.uniform(lp[0], lp[1])
                attn_layer_list = np.random.binomial(1, remain_prob, self.n_block).tolist()
                mlp_layer_list = np.random.binomial(1, remain_prob, self.n_block).tolist()

                for pass_layer in self.pass_layer_list:
                    blk, layer = pass_layer.split('.')
                    blk = int(blk)

                    if layer == 'self_attn':
                        attn_layer_list[blk] = 1
                    elif layer == 'mlp':
                        mlp_layer_list[blk] = 1
                    else:
                        raise NotImplementedError(f"layer : {layer}")
                    
                new_arch = {'layer': {'self_attn': attn_layer_list, 'mlp': mlp_layer_list}}
                complexity = get_net_info(new_arch, self.config, self.latency_table)
                # print(f'new_arch : {new_arch}')
                # print(f'complexity : {complexity}')
                if (new_arch not in data) and \
                    (new_arch not in pool) and \
                    (math.isclose(complexity[self.sec_obj], self.sec_obj_range[0]) or complexity[self.sec_obj] > self.sec_obj_range[0]) and \
                    (math.isclose(complexity[self.sec_obj], self.sec_obj_range[1]) or complexity[self.sec_obj] < self.sec_obj_range[1]) and \
                    (math.isclose(complexity['sparsity'], self.layer_prune_range[0]) or complexity['sparsity'] > self.layer_prune_range[0]) and \
                    (math.isclose(complexity['sparsity'], self.layer_prune_range[1]) or complexity['sparsity'] < self.layer_prune_range[1]) :
                    # print(f'selected arch : {complexity}')
                    # print('=' * 20)
                    break
                
            data.append(new_arch)
        return data

    def initialize(self, n_doe, pool=[]):
        # sample one arch with least (lb of hyperparameters) and most complexity (ub of hyperparameters)
        data = []
        # if math.isclose(self.sec_obj_range[0], min(self.quant_model_bits)):
        data.append(self.sample(lp=[self.layer_prune_range[1], self.layer_prune_range[1]])[0])
        n_doe -= 1
        # if math.isclose(self.sec_obj_range[-1], max(self.quant_model_bits)):
        data.append(self.sample(lp=[self.layer_prune_range[0], self.layer_prune_range[0]])[0])
        n_doe -= 1
        data.extend(self.sample(n_samples=n_doe, pool=pool))
        return data

    def encode(self, arch):
        # encode arch ({'q': [0, 2, 4], 'k: , etc}) to integer bit-string [1, 0, 2, 1, ...]
        attn_encode = np.array([np.argwhere(_x == np.array(self.layer_option))[0, 0] for _x in arch['layer']['self_attn']]).reshape(-1, 1)
        mlp_encode = np.array([np.argwhere(_x == np.array(self.layer_option))[0, 0] for _x in arch['layer']['mlp']]).reshape(-1, 1)

        return np.stack((attn_encode, mlp_encode), axis=-1).flatten()
    
    def decode(self, x):
        # decode integer bit-string [1, 0, 2, 1, ...] to arch ({'q': [0, 2, 4], 'k: , etc})
        x_reshape = x.reshape(self.n_block, self.n_layer)
        return {
                    'layer': {
                        'self_attn': np.array(self.layer_option)[x_reshape[:, 0]].tolist(),
                        'mlp': np.array(self.layer_option)[x_reshape[:, 1]].tolist()
                    }
                }


class LlamaMultiObjLayerSearchSpace:
    def __init__(self, 
                n_block,
                pass_layer_list=[],
                config=None,
                comp_obj='sparsity',
                comp_obj_min=[],
                comp_obj_max=[],
                # layer_prune_range=[],
                latency_table=None):
        self.n_block = n_block  # number of blocks

        self.layer_option = [0, 1]
        self.pass_layer_list = pass_layer_list
        self.config = config
        self.latency_table = latency_table
        self.n_layer = int(config['n_layer'])
        
        self.comp_obj = comp_obj
        self.comp_obj_min = comp_obj_min
        self.comp_obj_max = comp_obj_max
        # assert len(sec_obj_range) == 2, f"sec_obj_range is invalid: {sec_obj_range}"
        # assert math.isclose(sec_obj_range[0], sec_obj_range[1]) or sec_obj_range[0] < sec_obj_range[1], f"sec_obj_range is invalid: {sec_obj_range}"
        
        self.layer_prune_range = [0, 1]
        # assert len(layer_prune_range) == 2, f"layer_prune_range is invalid: {sec_obj_range}"
        # assert math.isclose(layer_prune_range[0], layer_prune_range[1]) or layer_prune_range[0] < layer_prune_range[1], f"layer_prune_range is invalid: {layer_prune_range}"

    def sample(self, n_samples=1, nb=None, lp=None, pool=[]):
        """ randomly sample a architecture"""
        nb = self.n_block if nb is None else nb
        lp = self.layer_prune_range if lp is None else lp
        
        data = []
        # for n in tqdm(range(n_samples), desc='Sampling'):
        for n in range(n_samples):
            while True:

                remain_prob = np.random.uniform(lp[0], lp[1])
                attn_layer_list = np.random.binomial(1, remain_prob, self.n_block).tolist()
                mlp_layer_list = np.random.binomial(1, remain_prob, self.n_block).tolist()

                # attn_layer_list = np.random.binomial(1, np.random.uniform(lp[0], lp[1]), self.n_block).tolist()
                # mlp_layer_list = np.random.binomial(1, np.random.uniform(lp[0], lp[1]), self.n_block).tolist()

                for pass_layer in self.pass_layer_list:
                    blk, layer = pass_layer.split('.')
                    blk = int(blk)

                    if layer == 'self_attn':
                        attn_layer_list[blk] = 1
                    elif layer == 'mlp':
                        mlp_layer_list[blk] = 1
                    else:
                        raise NotImplementedError(f"layer : {layer}")
                    
                new_arch = {'layer': {'self_attn': attn_layer_list, 'mlp': mlp_layer_list}}
                complexity = get_net_info(new_arch, self.config, self.latency_table)
                # print(f'new_arch : {new_arch}')
                # print(f'complexity : {complexity}')
                flag = (new_arch not in data) and (new_arch not in pool)
                for i, obj in enumerate(self.comp_obj):
                    flag &= (math.isclose(complexity[obj], self.comp_obj_min[i]) or complexity[obj] > self.comp_obj_min[i]) and \
                            (math.isclose(complexity[obj], self.comp_obj_max[1]) or complexity[obj] < self.comp_obj_max[i])
                if flag:
                    break

                # if (new_arch not in data) and \
                #     (new_arch not in pool) and \
                #     (math.isclose(complexity[self.sec_obj], self.sec_obj_range[0]) or complexity[self.sec_obj] > self.sec_obj_range[0]) and \
                #     (math.isclose(complexity[self.sec_obj], self.sec_obj_range[1]) or complexity[self.sec_obj] < self.sec_obj_range[1]) and \
                #     (math.isclose(complexity['sparsity'], self.layer_prune_range[0]) or complexity['sparsity'] > self.layer_prune_range[0]) and \
                #     (math.isclose(complexity['sparsity'], self.layer_prune_range[1]) or complexity['sparsity'] < self.layer_prune_range[1]) :
                #     # print(f'selected arch : {complexity}')
                #     # print('=' * 20)
                #     break
                
            data.append(new_arch)
        return data

    def initialize(self, n_doe, pool=[]):
        # sample one arch with least (lb of hyperparameters) and most complexity (ub of hyperparameters)
        data = []
        # if math.isclose(self.sec_obj_range[0], min(self.quant_model_bits)):
        data.append(self.sample(lp=[self.layer_prune_range[1], self.layer_prune_range[1]])[0])
        n_doe -= 1
        # if math.isclose(self.sec_obj_range[-1], max(self.quant_model_bits)):
        # data.append(self.sample(lp=[self.layer_prune_range[0], self.layer_prune_range[0]])[0])
        # n_doe -= 1
        data.extend(self.sample(n_samples=n_doe, pool=pool))
        return data

    def encode(self, arch):
        # encode arch ({'q': [0, 2, 4], 'k: , etc}) to integer bit-string [1, 0, 2, 1, ...]
        attn_encode = np.array([np.argwhere(_x == np.array(self.layer_option))[0, 0] for _x in arch['layer']['self_attn']]).reshape(-1, 1)
        mlp_encode = np.array([np.argwhere(_x == np.array(self.layer_option))[0, 0] for _x in arch['layer']['mlp']]).reshape(-1, 1)

        return np.stack((attn_encode, mlp_encode), axis=-1).flatten()
    
    def decode(self, x):
        # decode integer bit-string [1, 0, 2, 1, ...] to arch ({'q': [0, 2, 4], 'k: , etc})
        x_reshape = x.reshape(self.n_block, self.n_layer)
        return {
                    'layer': {
                        'self_attn': np.array(self.layer_option)[x_reshape[:, 0]].tolist(),
                        'mlp': np.array(self.layer_option)[x_reshape[:, 1]].tolist()
                    }
                }
    
class LlamaBlockSearchSpace:
    def __init__(self, 
                n_block,
                pass_layer_list=[],
                config=None,
                sec_obj='sparsity',
                sec_obj_range=[],
                layer_prune_range=[],
                latency_table=None):
        self.n_block = n_block  # number of blocks

        self.layer_option = [0, 1]
        self.pass_layer_list = pass_layer_list
        self.config = config
        self.latency_table = latency_table
        
        self.sec_obj = sec_obj
        self.sec_obj_range = sec_obj_range
        assert len(sec_obj_range) == 2, f"sec_obj_range is invalid: {sec_obj_range}"
        assert math.isclose(sec_obj_range[0], sec_obj_range[1]) or sec_obj_range[0] < sec_obj_range[1], f"sec_obj_range is invalid: {sec_obj_range}"
        
        self.layer_prune_range = layer_prune_range
        assert len(layer_prune_range) == 2, f"layer_prune_range is invalid: {sec_obj_range}"
        assert math.isclose(layer_prune_range[0], layer_prune_range[1]) or layer_prune_range[0] < layer_prune_range[1], f"layer_prune_range is invalid: {layer_prune_range}"

    def sample(self, n_samples=1, nb=None, lp=None, pool=[]):
        """ randomly sample a architecture"""
        nb = self.n_block if nb is None else nb
        lp = self.layer_prune_range if lp is None else lp
        
        data = []
        for n in tqdm(range(n_samples), desc='Sampling'):
            while True:

                remain_prob = np.random.uniform(lp[0], lp[1])
                block_list = np.random.binomial(1, remain_prob, self.n_block).tolist()

                for pass_layer in self.pass_layer_list:
                    blk, layer = pass_layer.split('.')
                    blk = int(blk)

                    if layer == 'self_attn':
                        block_list[blk] = 1
                    elif layer == 'mlp':
                        block_list[blk] = 1
                    else:
                        raise NotImplementedError(f"layer : {layer}")
                    
                new_arch = {'layer': {'self_attn': block_list, 'mlp': block_list}}
                complexity = get_net_info(new_arch, self.config, self.latency_table)
                # print(f'new_arch : {new_arch}')
                # print(f'complexity : {complexity}')
                if (new_arch not in data) and \
                    (new_arch not in pool) and \
                    (math.isclose(complexity[self.sec_obj], self.sec_obj_range[0]) or complexity[self.sec_obj] > self.sec_obj_range[0]) and \
                    (math.isclose(complexity[self.sec_obj], self.sec_obj_range[1]) or complexity[self.sec_obj] < self.sec_obj_range[1]) and \
                    (math.isclose(complexity['sparsity'], self.layer_prune_range[0]) or complexity['sparsity'] > self.layer_prune_range[0]) and \
                    (math.isclose(complexity['sparsity'], self.layer_prune_range[1]) or complexity['sparsity'] < self.layer_prune_range[1]) :
                    # print(f'selected arch : {complexity}')
                    # print('=' * 20)
                    break
                
            data.append(new_arch)
        return data

    def initialize(self, n_doe, pool=[]):
        # sample one arch with least (lb of hyperparameters) and most complexity (ub of hyperparameters)
        data = []
        # if math.isclose(self.sec_obj_range[0], min(self.quant_model_bits)):
        data.append(self.sample(lp=[self.layer_prune_range[1], self.layer_prune_range[1]])[0])
        n_doe -= 1
        # if math.isclose(self.sec_obj_range[-1], max(self.quant_model_bits)):
        data.append(self.sample(lp=[self.layer_prune_range[0], self.layer_prune_range[0]])[0])
        n_doe -= 1
        data.extend(self.sample(n_samples=n_doe, pool=pool))
        return data

    def encode(self, arch):
        # encode arch ({'q': [0, 2, 4], 'k: , etc}) to integer bit-string [1, 0, 2, 1, ...]
        return np.array([np.argwhere(_x == np.array(self.layer_option))[0, 0] for _x in arch['layer']['self_attn']])
    
    def decode(self, x):
        # decode integer bit-string [1, 0, 2, 1, ...] to arch ({'q': [0, 2, 4], 'k: , etc})
        return {
                    'layer': {
                        'self_attn': np.array(self.layer_option)[x].tolist(),
                        'mlp': np.array(self.layer_option)[x].tolist()
                    }
                }



# class LlamaLinearGroupSearchSpace:
#     def __init__(self,
#                 n_block,
#                 pass_linear_list=[],
#                 pass_layer_list=[],
#                 quant_model_bits=[],
#                 config=None,
#                 sec_obj='bits',
#                 sec_obj_range=[],
#                 layer_prune_range=[]):
#         self.n_block = n_block  # number of blocks

#         self.quant_model_bits = quant_model_bits
#         self.q_proj_option = self.k_proj_option = self.v_proj_option = quant_model_bits
#         self.o_proj_option = quant_model_bits
#         self.gate_proj_option = self.up_proj_option = quant_model_bits
#         self.down_proj_option = quant_model_bits

#         self.layer_option = [0, 1]
#         self.pass_linear_list = pass_linear_list
#         self.pass_layer_list = pass_layer_list
#         self.config = config
#         self.linear_group = ['self_attn.q_proj,self_attn.k_proj,self_attn.v_proj', 'self_attn.o_proj', 'mlp.gate_proj,mlp.up_proj', 'mlp.down_proj']
#         self.n_linear = len(self.linear_group)
#         self.n_layer = int(config['n_layer'])
        
#         self.sec_obj = sec_obj
#         self.sec_obj_range = sec_obj_range
#         assert len(sec_obj_range) == 2, f"sec_obj_range is invalid: {sec_obj_range}"
#         assert math.isclose(sec_obj_range[0], sec_obj_range[1]) or sec_obj_range[0] < sec_obj_range[1], f"sec_obj_range is invalid: {sec_obj_range}"
        
#         self.layer_prune_range = layer_prune_range
#         assert len(layer_prune_range) == 2, f"layer_prune_range is invalid: {sec_obj_range}"
#         assert math.isclose(layer_prune_range[0], layer_prune_range[1]) or layer_prune_range[0] < layer_prune_range[1], f"layer_prune_range is invalid: {layer_prune_range}"

#         self.n_bits = len(quant_model_bits)

#     def sample(self, n_samples=1, nb=None, q=None, k=None, v=None, o=None, down=None, up=None, gate=None, lp=None, pool=[]):
#         """ randomly sample a architecture"""
#         nb = self.n_block if nb is None else nb
#         q = self.q_proj_option if q is None else q
#         k = self.k_proj_option if k is None else k
#         v = self.v_proj_option if v is None else v
#         o = self.o_proj_option if o is None else o
#         gate = self.gate_proj_option if gate is None else gate
#         up = self.up_proj_option if up is None else up
#         down = self.down_proj_option if down is None else down
#         lp = self.layer_prune_range if lp is None else lp
        
#         data = []
#         for n in tqdm(range(n_samples), desc='Sampling'):
#             while True:
#                 prob = np.random.rand(3)
#                 # q_prob = np.random.rand(len(q))
#                 qkv_prob = prob[np.array([np.argwhere(_x == np.array(self.q_proj_option))[0, 0] for _x in q])]
#                 qkv_list = np.random.choice(q, size=nb, p=qkv_prob / qkv_prob.sum(), replace=True).tolist()
                
#                 # o_prob = np.random.rand(len(o))
#                 o_prob = prob[np.array([np.argwhere(_x == np.array(self.o_proj_option))[0, 0] for _x in o])]
#                 o_list = np.random.choice(o, size=nb, p=o_prob / o_prob.sum(), replace=True).tolist()

#                 # gate_prob = np.random.rand(len(gate))
#                 gateup_prob = prob[np.array([np.argwhere(_x == np.array(self.gate_proj_option))[0, 0] for _x in gate])]
#                 gateup_list = np.random.choice(gate, size=nb, p=gateup_prob / gateup_prob.sum(), replace=True).tolist()

#                 # down_prob = np.random.rand(len(down))
#                 down_prob = prob[np.array([np.argwhere(_x == np.array(self.down_proj_option))[0, 0] for _x in down])]
#                 down_list = np.random.choice(down, size=nb, p=down_prob / down_prob.sum(), replace=True).tolist()

#                 remain_prob = np.random.uniform(lp[0], lp[1])
#                 attn_layer_list = np.random.binomial(1, remain_prob, self.n_block).tolist()
#                 mlp_layer_list = np.random.binomial(1, remain_prob, self.n_block).tolist()

#                 for pass_linear in self.pass_linear_list:
#                     blk, linear = pass_linear.split('.')[0], pass_linear.split('.')[-1]
#                     blk = int(blk)
#                     if linear == 'q_proj' or linear == 'k_proj' or linear == 'v_proj':
#                         qkv_list[blk] = max(self.q_proj_option)
#                         attn_layer_list[blk] = 1
#                     elif linear == 'o_proj':
#                         o_list[blk] = max(self.o_proj_option)
#                         attn_layer_list[blk] = 1
                        
#                     elif linear == 'gate_proj' or linear == 'up_proj':
#                         gateup_list[blk] = max(self.gate_proj_option)
#                         mlp_layer_list[blk] = 1
#                     elif linear == 'down_proj':
#                         down_list[blk] = max(self.down_proj_option)
#                         mlp_layer_list[blk] = 1
#                     else:
#                         raise NotImplementedError(f"linear : {linear}")

#                 for pass_layer in self.pass_layer_list:
#                     blk, layer = pass_layer.split('.')
#                     blk = int(blk)

#                     if layer == 'self_attn':
#                         attn_layer_list[blk] = 1
#                     elif layer == 'mlp':
#                         mlp_layer_list[blk] = 1
#                     else:
#                         raise NotImplementedError(f"layer : {layer}")
                    
#                 new_arch = {'linear': {'self_attn.q_proj': qkv_list, 'self_attn.k_proj': qkv_list, 'self_attn.v_proj': qkv_list, 'self_attn.o_proj': o_list, 'mlp.gate_proj': gateup_list, 'mlp.up_proj': gateup_list, 'mlp.down_proj': down_list}, 'layer': {'self_attn': attn_layer_list, 'mlp': mlp_layer_list}}
#                 complexity = get_net_info(new_arch, self.config)
#                 # print(f'new_arch : {new_arch}')
#                 # print(f'complexity : {complexity}')
#                 if (new_arch not in data) and \
#                     (new_arch not in pool) and \
#                     (math.isclose(complexity[self.sec_obj], self.sec_obj_range[0]) or complexity[self.sec_obj] > self.sec_obj_range[0]) and \
#                     (math.isclose(complexity[self.sec_obj], self.sec_obj_range[1]) or complexity[self.sec_obj] < self.sec_obj_range[1]) and \
#                     (math.isclose(complexity['sparsity'], self.layer_prune_range[0]) or complexity['sparsity'] > self.layer_prune_range[0]) and \
#                     (math.isclose(complexity['sparsity'], self.layer_prune_range[1]) or complexity['sparsity'] < self.layer_prune_range[1]) :
#                     # print(f'selected arch : {complexity} bits')
#                     # print('=' * 20)
#                     break
                
#             data.append(new_arch)
#         return data

#     def initialize(self, n_doe, pool=[]):
#         # sample one arch with least (lb of hyperparameters) and most complexity (ub of hyperparameters)
#         data = []
#         if math.isclose(self.sec_obj_range[0], min(self.quant_model_bits)):
#             data.append(self.sample(q=[min(self.quant_model_bits)], k=[min(self.quant_model_bits)], v=[min(self.quant_model_bits)], o=[min(self.quant_model_bits)], down=[min(self.quant_model_bits)], up=[min(self.quant_model_bits)], gate=[min(self.quant_model_bits)], lp=[1, 1])[0])
#             n_doe -= 1
#         if math.isclose(self.sec_obj_range[-1], max(self.quant_model_bits)):
#             data.append(self.sample(q=[max(self.quant_model_bits)], k=[max(self.quant_model_bits)], v=[max(self.quant_model_bits)], o=[max(self.quant_model_bits)], down=[max(self.quant_model_bits)], up=[max(self.quant_model_bits)], gate=[max(self.quant_model_bits)], lp=[1, 1])[0])
#             n_doe -= 1
#         data.extend(self.sample(n_samples=n_doe, pool=pool))
#         return data

#     def encode(self, arch):
#         # encode arch ({'q': [0, 2, 4], 'k: , etc}) to integer bit-string [1, 0, 2, 1, ...]
#         qkv_encode = np.array([np.argwhere(_x == np.array(self.q_proj_option))[0, 0] for _x in arch['linear']['self_attn.q_proj']]).reshape(-1, 1)
#         o_encode = np.array([np.argwhere(_x == np.array(self.o_proj_option))[0, 0] for _x in arch['linear']['self_attn.o_proj']]).reshape(-1, 1)
#         gateup_encode = np.array([np.argwhere(_x == np.array(self.gate_proj_option))[0, 0] for _x in arch['linear']['mlp.gate_proj']]).reshape(-1, 1)
#         down_encode = np.array([np.argwhere(_x == np.array(self.down_proj_option))[0, 0] for _x in arch['linear']['mlp.down_proj']]).reshape(-1, 1)
#         attn_encode = np.array([np.argwhere(_x == np.array(self.layer_option))[0, 0] for _x in arch['layer']['self_attn']]).reshape(-1, 1)
#         mlp_encode = np.array([np.argwhere(_x == np.array(self.layer_option))[0, 0] for _x in arch['layer']['mlp']]).reshape(-1, 1)

#         return np.stack((qkv_encode, o_encode, gateup_encode, down_encode, attn_encode, mlp_encode), axis=-1).reshape(-1)

#     def decode(self, x):
#         # decode integer bit-string [1, 0, 2, 1, ...] to arch ({'q': [0, 2, 4], 'k: , etc})
#         x_reshape = x.reshape(self.n_block, self.n_linear + self.n_layer)
#         return {
#                     'linear': {
#                         'self_attn.q_proj': np.array(self.q_proj_option)[x_reshape[:, 0]].tolist(),
#                         'self_attn.k_proj': np.array(self.k_proj_option)[x_reshape[:, 0]].tolist(),
#                         'self_attn.v_proj': np.array(self.v_proj_option)[x_reshape[:, 0]].tolist(),
#                         'self_attn.o_proj': np.array(self.o_proj_option)[x_reshape[:, 1]].tolist(),
#                         'mlp.gate_proj': np.array(self.gate_proj_option)[x_reshape[:, 2]].tolist(),
#                         'mlp.up_proj': np.array(self.up_proj_option)[x_reshape[:, 2]].tolist(),
#                         'mlp.down_proj': np.array(self.down_proj_option)[x_reshape[:, 3]].tolist(),
#                     },
#                     'layer': {
#                         'self_attn': np.array(self.layer_option)[x_reshape[:, 4]].tolist(),
#                         'mlp': np.array(self.layer_option)[x_reshape[:, 5]].tolist()
#                     }
#                 }
