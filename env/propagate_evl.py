import numpy as np
from typing import Union
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size

import torch
from torch import Tensor
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing


class ForwardPass(MessagePassing):
    def __init__(self, **kwargs):
        kwargs.setdefault('aggr', 'max')
        super(ForwardPass, self).__init__(**kwargs)

    def forward(self,
                x: Union[Tensor, OptPairTensor],
                edge_index: Adj,
                size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)
        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size)
        return out


class BackwardPass(MessagePassing):
    def __init__(self, **kwargs):
        kwargs.setdefault('aggr', 'max')
        super(BackwardPass, self).__init__(**kwargs)

    def forward(self,
                x: Union[Tensor, OptPairTensor],
                edge_index: Adj,
                size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)
        # print(x)
        # print(edge_index)
        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size)
        # print(out)
        return out


if __name__ == "__main__":
    from generateJSP import uni_instance_gen
    from env.env_single import JsspN5
    import time

    j = 100
    m = 20
    l = 1
    h = 99
    np.random.seed(3)

    env = JsspN5(n_job=j, n_mch=m, low=l, high=h, init='rule', rule='fdd/mwkr', transition=0)
    inst = np.concatenate([uni_instance_gen(n_j=j, n_m=m, low=l, high=h)])

    t1 = time.time()
    state, feasible_action, done = env.reset(instance=inst, fix_instance=True)
    t2 = time.time()

    # testing forward pass
    dur_earliest_st = torch.from_numpy(np.pad(inst[0].reshape(-1), (1, 1), 'constant', constant_values=0)).reshape(-1, 1)
    forward_pass = ForwardPass(aggr='max', flow="source_to_target")
    earliest_st = torch.zeros(size=[j * m + 2, 1], dtype=torch.float32)
    adj_earliest_st = state.edge_index[:, state.edge_index[0] != state.edge_index[1]]
    ma_earliest_st = torch.ones(size=[j * m + 2, 1], dtype=torch.int8)
    ma_earliest_st[0] = 0

    t3 = time.time()
    for _ in range(j*m+2):
        if ma_earliest_st.sum() == 0:
            print('finish forward pass at step:', _)
            break
        x = dur_earliest_st + earliest_st.masked_fill(ma_earliest_st, 0)
        earliest_st = forward_pass(x=x, edge_index=adj_earliest_st)
        ma_earliest_st = forward_pass(x=ma_earliest_st, edge_index=adj_earliest_st)
    t4 = time.time()
    if torch.equal(earliest_st.squeeze() / 1000, state.x[:, 1]):
        print('forward pass is OK! It takes:', t4 - t3, 'networkx version forward pass and backward pass take:', t2 - t1)

    print()

    # testing backward pass
    dur_latest_st = torch.from_numpy(np.pad(inst[0].reshape(-1), (1, 1), 'constant', constant_values=0)).reshape(-1, 1)
    backward_pass = BackwardPass(aggr='max', flow="target_to_source")
    latest_st = torch.zeros(size=[j * m + 2, 1], dtype=torch.float32)
    latest_st[-1] = - float(state.y)
    adj_latest_st = state.edge_index[:, state.edge_index[0] != state.edge_index[1]]
    ma_latest_st = torch.ones(size=[j * m + 2, 1], dtype=torch.int8)
    ma_latest_st[-1] = 0
    t3 = time.time()
    for _ in range(j * m + 2):  # j * m + 2
        if ma_latest_st.sum() == 0:
            print('finish backward pass at step:', _)
            break
        x = latest_st.masked_fill(ma_latest_st, 0)
        latest_st = backward_pass(x=x, edge_index=adj_latest_st) + dur_latest_st
        latest_st[-1] = - float(state.y)
        ma_latest_st = backward_pass(x=ma_latest_st, edge_index=adj_latest_st)
    t4 = time.time()
    if torch.equal(- latest_st.squeeze() / 1000, state.x[:, 2]):
        print('backward pass is OK! It takes:', t4 - t3, 'networkx version forward pass and backward pass take:', t2 - t1)