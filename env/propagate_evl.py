import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from typing import Union
from torch_geometric.typing import OptPairTensor, Adj, Size
import torch
from torch import Tensor
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
        out = self.propagate(edge_index, x=x, size=size)
        return out


class Evaluator:
    def __init__(self):
        self.forward_pass = ForwardPass(aggr='max', flow="source_to_target")
        self.backward_pass = BackwardPass(aggr='max', flow="target_to_source")

    def forward(self, edge_index, duration, n_j, n_m):
        """
        support batch version
        edge_index: [2, n_edges] tensor
        duration: [n_nodes, 1] tensor
        """
        n_nodes = duration.shape[0]
        n_nodes_each_graph = n_j * n_m + 2
        device = edge_index.device

        # forward pass...
        index_S = np.arange(n_nodes // n_nodes_each_graph, dtype=int) * n_nodes_each_graph
        earliest_start_time = torch.zeros_like(duration, dtype=torch.float32, device=device)
        mask_earliest_start_time = torch.ones_like(duration, dtype=torch.int8, device=device)
        mask_earliest_start_time[index_S] = 0
        for _ in range(n_nodes):
            if mask_earliest_start_time.sum() == 0:
                break
            x_forward = duration + earliest_start_time.masked_fill(mask_earliest_start_time.bool(), 0)
            earliest_start_time = self.forward_pass(x=x_forward, edge_index=edge_index)
            mask_earliest_start_time = self.forward_pass(x=mask_earliest_start_time, edge_index=edge_index)

        # backward pass...
        index_T = np.cumsum(np.ones(shape=[n_nodes // n_nodes_each_graph], dtype=int) * n_nodes_each_graph) - 1
        make_span = earliest_start_time[index_T]
        latest_start_time = torch.zeros_like(duration, dtype=torch.float32, device=device)
        latest_start_time[index_T] = - make_span
        mask_latest_start_time = torch.ones_like(duration, dtype=torch.int8, device=device)
        mask_latest_start_time[index_T] = 0
        for _ in range(n_nodes):
            if mask_latest_start_time.sum() == 0:
                break
            x_backward = latest_start_time.masked_fill(mask_latest_start_time.bool(), 0)
            latest_start_time = self.backward_pass(x=x_backward, edge_index=edge_index) + duration
            latest_start_time[index_T] = - make_span
            mask_latest_start_time = self.backward_pass(x=mask_latest_start_time, edge_index=edge_index)

        return earliest_start_time, torch.abs(latest_start_time), make_span


if __name__ == "__main__":
    from generateJSP import uni_instance_gen
    from env.env_single import JsspN5
    import time
    from torch_geometric.data.batch import Batch

    j = 30
    m = 20
    l = 1
    h = 99
    batch_size = 64
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    np.random.seed(1)

    env = JsspN5(n_job=j, n_mch=m, low=l, high=h, init='rule', rule='fdd-divide-mwkr', transition=0)
    insts = [np.concatenate([uni_instance_gen(n_j=j, n_m=m, low=l, high=h)]) for _ in range(batch_size)]

    t1 = time.time()
    state, feasible_action, done = env.reset(instance=insts[0], fix_instance=True)
    for _ in range(1, batch_size):
        state, feasible_action, done = env.reset(instance=insts[_], fix_instance=True)
    t2 = time.time()

    # testing forward pass for first instance
    inst = insts[-1]
    state, feasible_action, done = env.reset(instance=inst, fix_instance=True)
    dur_earliest_st = torch.from_numpy(np.pad(inst[0].reshape(-1), (1, 1), 'constant', constant_values=0)).reshape(-1, 1).to(dev)
    forward_pass = ForwardPass(aggr='max', flow="source_to_target")
    earliest_st = torch.zeros(size=[j * m + 2, 1], dtype=torch.float32, device=dev)
    adj_earliest_st = state.edge_index[:, state.edge_index[0] != state.edge_index[1]].to(dev)
    ma_earliest_st = torch.ones(size=[j * m + 2, 1], dtype=torch.int8, device=dev)
    ma_earliest_st[0] = 0

    t3 = time.time()
    for _ in range(j*m+2):
        if ma_earliest_st.sum() == 0:
            print('finish forward pass at step:', _)
            break
        x = dur_earliest_st + earliest_st.masked_fill(ma_earliest_st.bool(), 0)
        earliest_st = forward_pass(x=x, edge_index=adj_earliest_st)
        ma_earliest_st = forward_pass(x=ma_earliest_st, edge_index=adj_earliest_st)
    t4 = time.time()
    if torch.equal(earliest_st.cpu().squeeze() / 1000, state.x[:, 1]):
        print('forward pass is OK! It takes:', t4 - t3, 'networkx version forward pass and backward pass take:', t2 - t1)

    print()

    # testing backward pass for first instance
    inst = insts[-1]
    state, feasible_action, done = env.reset(instance=inst, fix_instance=True)
    dur_latest_st = torch.from_numpy(np.pad(inst[0].reshape(-1), (1, 1), 'constant', constant_values=0)).reshape(-1, 1).to(dev)
    backward_pass = BackwardPass(aggr='max', flow="target_to_source")
    latest_st = torch.zeros(size=[j * m + 2, 1], dtype=torch.float32, device=dev)
    latest_st[-1] = - float(state.y)
    adj_latest_st = state.edge_index[:, state.edge_index[0] != state.edge_index[1]].to(dev)
    ma_latest_st = torch.ones(size=[j * m + 2, 1], dtype=torch.int8, device=dev)
    ma_latest_st[-1] = 0
    t3 = time.time()
    for _ in range(j * m + 2):  # j * m + 2
        if ma_latest_st.sum() == 0:
            print('finish backward pass at step:', _)
            break
        x = latest_st.masked_fill(ma_latest_st.bool(), 0)
        latest_st = backward_pass(x=x, edge_index=adj_latest_st) + dur_latest_st
        latest_st[-1] = - float(state.y)
        ma_latest_st = backward_pass(x=ma_latest_st, edge_index=adj_latest_st)
    t4 = time.time()
    if torch.equal(- latest_st.squeeze().cpu() / 1000, state.x[:, 2]):
        print('backward pass is OK! It takes:', t4 - t3, 'networkx version forward pass and backward pass take:', t2 - t1)

    print()

    # test hybrid evaluator
    state_list = []
    dur_list = []
    for _ in range(batch_size):
        state, feasible_action, done = env.reset(instance=insts[_], fix_instance=True)
        state_list.append(state)
        dur_list.append(np.pad(insts[_][0].reshape(-1), (1, 1), 'constant', constant_values=0))
    batch_data = Batch.from_data_list(state_list)
    edge_idx = batch_data.edge_index[:, batch_data.edge_index[0] != batch_data.edge_index[1]].to(dev)
    dur = np.concatenate(dur_list)
    dur = torch.from_numpy(dur).reshape(-1, 1).to(dev)
    eva = Evaluator()
    t5 = time.time()
    est, lst, _ = eva.forward(edge_index=edge_idx, duration=dur, n_j=j, n_m=m)
    t6 = time.time()
    if torch.equal(est.cpu().squeeze() / 1000, batch_data.x[:, 1]) and torch.equal(lst.squeeze().cpu() / 1000, batch_data.x[:, 2]):
        print('forward pass and backward pass are all OK! It takes:', t6 - t5, 'networkx version forward pass and backward pass take:', t2 - t1)

