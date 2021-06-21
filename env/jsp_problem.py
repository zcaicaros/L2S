import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def forward_pass(graph, topological_order):  # graph is a nx.DiGraph;
    # assert (graph.in_degree(topological_order[0]) == 0)
    earliest_ST = dict.fromkeys(graph.nodes, -float('inf'))
    earliest_ST[topological_order[0]] = 0.
    # topo_order = nx.topological_sort(graph)
    for n in topological_order:
        for s in graph.successors(n):
            if earliest_ST[s] < earliest_ST[n] + graph.edges[n, s]['weight']:
                earliest_ST[s] = earliest_ST[n] + graph.edges[n, s]['weight']
    # return is a dict where key is each node's ID, value is the length from source node s
    return earliest_ST


def backward_pass(graph, topological_order, makespan):
    reverse_order = list(reversed(topological_order))
    latest_ST = dict.fromkeys(graph.nodes, float('inf'))
    latest_ST[reverse_order[0]] = float(makespan)
    for n in reverse_order:
        for p in graph.predecessors(n):
            if latest_ST[p] > latest_ST[n] - graph.edges[p, n]['weight']:
                # assert latest_ST[n] - graph.edges[p, n]['weight'] >= 0, 'latest start times should is negative, BUG!'  # latest start times should be non-negative
                latest_ST[p] = latest_ST[n] - graph.edges[p, n]['weight']
    return latest_ST


def mat2graph(adj_mat, dur_mat, plot_G=False):
    '''
    adj_mat: the same adj from our NeurIPS 2020 paper
    dur_mat: the same dur from our NeurIPS 2020 paper
    '''
    # prepare adj and dur
    n_job = dur_mat.shape[0]
    n_mch = dur_mat.shape[1]
    # pad dummy S and T nodes
    adj_mat = np.pad(adj_mat, 1, 'constant', constant_values=0)
    # connect S with 1st operation of each job
    adj_mat[[i for i in range(1, n_job * n_mch + 2 - 1, n_mch)], 0] = 1
    # connect last operation of each job to T
    adj_mat[-1, [i for i in range(n_mch, n_job * n_mch + 2 - 1, n_mch)]] = 1
    adj_mat = np.rot90(np.fliplr(adj_mat))  # convert input adj from column pointing to row, to, row pointing to column
    dur_mat = np.pad(dur_mat.reshape(-1, 1), ((1, 1), (0, 0)), 'constant', constant_values=0).repeat(n_job * n_mch + 2, axis=1)
    edge_weight = np.multiply(adj_mat, dur_mat)
    # create nx.DiGraph
    G = nx.from_numpy_matrix(edge_weight, parallel_edges=False, create_using=nx.DiGraph)
    G.add_weighted_edges_from([(0, i, 0) for i in range(1, n_job * n_mch + 2 - 1, n_mch)])  # add release time, here all jobs are available at t=0. This is the only way to add release date. And if you do not add release date, startime computation will return wired value
    # add self-loop
    np.fill_diagonal(adj_mat, val=1)

    if plot_G:
        x_axis = np.pad(np.tile(np.arange(1, n_mch + 1, 1), n_job), (1, 1), 'constant',
                        constant_values=[0, n_mch + 1])
        y_axis = np.pad(np.arange(n_job, 0, -1).repeat(n_mch), (1, 1), 'constant',
                        constant_values=np.median(np.arange(n_job, 0, -1)))
        pos = dict((n, (x, y)) for n, x, y in zip(G.nodes(), x_axis, y_axis))
        plt.figure(figsize=(15, 10))
        plt.tight_layout()
        nx.draw_networkx_edge_labels(G, pos=pos)  # show edge weight
        nx.draw(
            G, pos=pos, with_labels=True, arrows=True, connectionstyle='arc3, rad = 0.1'  # <-- tune curvature and style ref:https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.patches.ConnectionStyle.html
        )
        plt.show()

    return G, adj_mat  # adj_mat: self-loop added, S&T included


def forward_and_backward_pass(adj_mat, dur_mat, plot_G=False):
    '''
    adj_mat: the same adj from our NeurIPS 2020 paper
    dur_mat: the same dur from our NeurIPS 2020 paper
    '''
    G, adj_augmented = mat2graph(adj_mat=adj_mat, dur_mat=dur_mat, plot_G=plot_G)

    # calculate topological order
    topological_order = list(nx.topological_sort(G))
    # forward and backward pass
    est_ST = np.fromiter(forward_pass(graph=G, topological_order=topological_order).values(), dtype=float)
    lst_ST = np.fromiter(backward_pass(graph=G, topological_order=topological_order, makespan=est_ST[-1]).values(), dtype=float)
    # assert np.where(est_ST > lst_ST)[0].shape[0] == 0, 'latest starting time is smaller than earliest starting time, bug!'  # latest starting time should be larger or equal to earliest starting time
    return est_ST, lst_ST, adj_augmented, G


def eval_priority_list(p_list, dur_mat, mch_mat, plot=False):

    adj_mat = list2simpleAdj(p_list=p_list, mch_mat=mch_mat)
    # forward and backward paths
    earliest_st, latest_st, adj_mat_aug, G = forward_and_backward_pass(adj_mat, dur_mat, plot_G=plot)
    return np.amax(earliest_st, axis=0), earliest_st, latest_st, adj_mat_aug, G


def list2simpleAdj(p_list, mch_mat):
    no_jobs = mch_mat.shape[0]
    no_machines = mch_mat.shape[1]
    no_operations = no_jobs * no_machines
    # Init operations mat
    ops_mat = np.arange(0, no_operations).reshape(mch_mat.shape).tolist()
    # Init list_for_latest_task_onMachine
    list_for_latest_task_onMachine = [None] * no_machines
    # Create adjacent matrix for the corresponding action list
    adj_mat = np.eye(no_operations, k=-1, dtype=int)
    adj_mat[np.arange(start=0, stop=no_operations, step=1).reshape(no_jobs, -1)[:,
            0]] = 0  # first column does not have upper stream conj_nei
    # Construct adjacent matrix
    for job_id in p_list:
        op_id = ops_mat[job_id][0]
        m_id_for_action = mch_mat[op_id // no_machines, op_id % no_machines] - 1
        if list_for_latest_task_onMachine[m_id_for_action] is not None:
            adj_mat[op_id, list_for_latest_task_onMachine[m_id_for_action]] = 1
        list_for_latest_task_onMachine[m_id_for_action] = op_id
        ops_mat[job_id].pop(0)
    return adj_mat  # no self-loop


if __name__ == '__main__':
    from generateJSP import uni_instance_gen
    import random
    import time
    # random.seed(1)
    # np.random.seed(1)

    nj = 100
    nm = 20
    low = 1  # The lower bound of processing time
    high = 99  # The upper bound of processing time, processing time is uniformly sampled from: U[low, high]

    # generate JSSP instance
    dur, mch = uni_instance_gen(n_j=nj, n_m=nm, low=low, high=high)
    # print('Precedent constraints:\n', mch)
    # print('Processing time:\n', dur)

    # create random priority list
    priority_list = [i for i in range(nj) for m in range(nm)]
    random.shuffle(priority_list)
    # print('Priority list is:', priority_list)

    t1 = time.time()
    make_span, _, _, adj_aug, _ = eval_priority_list(p_list=priority_list, dur_mat=dur, mch_mat=mch, plot=False)
    t2 = time.time()
    print('Computation time is:', t2 - t1)
    print('The makespan is:', make_span)
    # print('The adjacent matrix is:\n', adj)




