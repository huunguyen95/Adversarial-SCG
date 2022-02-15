import os
import json
import pickle
import numpy as np
import pandas as pd
import networkx as nx
from joblib import load
from glob import glob
from copy import deepcopy
from Graph2vec import graph2vec


def update(num_episode, env, RL):
    avail_action = env.get_avail_action()
    if not avail_action:
        return False

    for e in range(num_episode):
        step = 0
        state = env.reset()

        while step < 20:
            action = RL.choose_action(str(state), avail_action)

            new_state, reward, done = env.step(action, state, step)

            RL.learn(str(state), action, reward, str(new_state))
            state = dict(new_state)

            if done:
                break
            step += 1
        if done:
            print(state)
            break
        # print(e, state, done)
    return done


def RL(path):
    base = os.path.basename(path)
    gname = os.path.splitext(base)[0]
    G = nx.read_adjlist(path, create_using=nx.DiGraph)

    edges = deepcopy(G.edges)
    for (u, v) in edges:
        if "execve" in u:
            G.add_edge(u, "fork")
            G.add_edge("fork", v)
            G.remove_edge(u, v)

    beg_path = np.random.choice(glob("data/origin/benign/*.adjlist"))
    H = nx.read_adjlist(beg_path, create_using=nx.DiGraph)

    mapping = {}
    for n in H:
        mapping[n] = f"beg_{n}"
    H = nx.relabel_nodes(H, mapping=mapping)

    for (u, v) in H.edges:
        G.add_edge(u, v)
        if "execve" in u:
            G.add_edge("fork", u)
    G = nx.relabel_nodes(G, mapping=dict(zip(G, map(str, range(len(G))))))
    print(G.edges)

    env = Environment(graph=G, label=1, gname=gname)
    RL = QLearningTable(actions=list(H.nodes))

    n_eps = 20
    done = update(n_eps, env, RL)
    if not done:
        print("Attack failed.")


PATH = glob("data/origin/malware/*.adjlist")
for path in PATH:
    RL(path)
    exit()
