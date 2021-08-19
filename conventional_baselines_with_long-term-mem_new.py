from env.env_batch import JsspN5
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def show_state(G, j, m):
    x_axis = np.pad(np.tile(np.arange(1, m + 1, 1), j), (1, 1), 'constant', constant_values=[0, m + 1])
    y_axis = np.pad(np.arange(j, 0, -1).repeat(m), (1, 1), 'constant', constant_values=np.median(np.arange(j, 0, -1)))
    pos = dict((n, (x, y)) for n, x, y in zip(G.nodes(), x_axis, y_axis))
    plt.figure(figsize=(15, 10))
    plt.tight_layout()
    nx.draw_networkx_edge_labels(G, pos=pos)  # show edge weight
    nx.draw(
        G, pos=pos, with_labels=True, arrows=True, connectionstyle='arc3, rad = 0.1'
        # <-- tune curvature and style ref:https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.patches.ConnectionStyle.html
    )
    plt.show()


def change_nxgraph_topology(self, action, G, instance, plot=False):
    n_jobs, n_machines = self.instances[0].shape
    n_operations = n_jobs * n_machines

    if action == [0, 0]:  # if dummy action then do nothing
        pass
    else:  # change nx graph topology
        S = [s for s in G.predecessors(action[0]) if
             int((s - 1) // n_machines) != int((action[0] - 1) // n_machines) and s != 0]
        T = [t for t in G.successors(action[1]) if
             int((t - 1) // n_machines) != int((action[1] - 1) // n_machines) and t != n_operations + 1]
        s = S[0] if len(S) != 0 else None
        t = T[0] if len(T) != 0 else None

        if s is not None:  # connect s with action[1]
            G.remove_edge(s, action[0])
            G.add_edge(s, action[1], weight=np.take(instance[0], s - 1))
        else:
            pass

        if t is not None:  # connect action[0] with t
            G.remove_edge(action[1], t)
            G.add_edge(action[0], t, weight=np.take(instance[0], action[0] - 1))
        else:
            pass

        # reverse edge connecting selected pair
        G.remove_edge(action[0], action[1])
        G.add_edge(action[1], action[0], weight=np.take(instance[0], action[1] - 1))

    if plot:
        self.show_state(G)