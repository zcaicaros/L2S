import numpy as np
from typing import Union
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size

import torch
from torch import Tensor
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing


class EVL(MessagePassing):
    def __init__(self, **kwargs):
        kwargs.setdefault('aggr', 'max')
        super(EVL, self).__init__(**kwargs)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size)

        return out


if __name__ == "__main__":
    j = 3
    m = 3
    l = 1
    h = 99
    evaluator = EVL(aggr='max', flow="source_to_target")
    st = torch.zeros(size=[j*m+2])
    dur = torch.from_numpy(np.pad(np.random.randint(low=l, high=h, size=j*m, dtype=int), (1, 1), 'constant', constant_values=0).astype(float))
    adj = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5,
                         6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 10],
                        [0, 1, 4, 7, 1, 2, 9, 2, 3, 6, 3, 10, 1, 4, 5, 5, 6, 8,
                         6, 10, 2, 7, 8, 3, 8, 9, 9, 10, 10]])
    print(st.shape)
    print(dur.shape)
    print(adj.shape)

    for _ in range(j*m+2):
        x = dur.reshape(j*m+2, 1) + st.reshape(j*m+2, 1)
        st = evaluator(x=x, edge_index=adj)
    print(st)