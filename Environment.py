import networkx as nx
from copy import deepcopy
from joblib import load


class Environment():
    def __init__(self, graph, graph_name):
        self.init_graph = graph
        self.graph = deepcopy(self.init_graph)
        self.graph_name = graph_name
        self.init_state = dict()

        self.avail_action = []
        for node in self.graph:
            if node.startswith('beg'):
                self.avail_action.append(node)

        self.scaler = load("model/scaler.joblib")
        self.clf = load("model/RF.joblib")

    def reset(self):
        self.graph = deepcopy(self.init_graph)
        return self.init_state

    def get_avail_action(self):
        return self.avail_action

    def step(self, action, state, step):
        reward = -1
        # dummy = action + '+' + str(state.get(action, 0) + 1)
        dummy = action + '+'
        if dummy not in self.graph.nodes:
            self.graph.add_node(dummy)
        self.graph.add_edge(action, dummy)
        node = dict()
        G = nx.DiGraph()
        for e in self.graph.edges:
            u, v = e
            if u not in node:
                node[u] = len(node) + 1
            if v not in node:
                node[v] = len(node) + 1

            G.add_edge(node[u], node[v])

        # nx.write_adjlist(G, "data/origin/test/adversarial.adjlist")
        # graph2vec()
        # data = pd.read_csv("data/features.csv")
        # x = data.loc[data["type"] == "adversarial", "x_0":].values
        # x = self.scaler.transform(x)
        # y = self.clf.predict(x)[0]
        y = 0

        if step == 49:
            reward = -1000
        if y == 1 - self.label:
            reward = 1000
            done = True
        else:
            done = False

        next_state = dict(state)
        next_state[action] = next_state.get(action, 0) + 1
        return next_state, reward, done
