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
        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size)
        return out


if __name__ == "__main__":
    from generateJSP import uni_instance_gen
    from env.env_single import JsspN5
    import time

    j = 15
    m = 15
    l = 1
    h = 99
    np.random.seed(3)

    env = JsspN5(n_job=j, n_mch=m, low=l, high=h, init='rule', rule='fdd/mwkr', transition=0)
    inst = np.concatenate([uni_instance_gen(n_j=j, n_m=m, low=l, high=h)])

    t1 = time.time()
    state, feasible_action, done = env.reset(instance=inst, fix_instance=True)
    t2 = time.time()

    evaluator = BackwardPass(aggr='max', flow="source_to_target")
    st = torch.zeros(size=[j * m + 2, 1], dtype=torch.float32)
    dur = torch.from_numpy(np.pad(inst[0].reshape(-1), (1, 1), 'constant', constant_values=0)).reshape(-1, 1)
    adj = state.edge_index[:, state.edge_index[0] != state.edge_index[1]]
    ma = torch.ones(size=[j * m + 2, 1], dtype=torch.int8)
    ma[0] = 0

    t3 = time.time()
    for _ in range(j*m+2):
        if ma.sum() == 0:
            print(_)
            break
        x = dur + st.masked_fill(ma, 0)
        st = evaluator(x=x, edge_index=adj)
        ma = evaluator(x=ma, edge_index=adj)
    t4 = time.time()
    if torch.equal(st.squeeze()/1000, state.x[:, 1]):
        print('yes')
    print(t2 - t1)
    print(t4 - t3)