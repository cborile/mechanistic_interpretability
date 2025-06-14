import numpy as np
import pickle


def get_ranking_dict(scores, scores1, topk):
    n0 = [np.unravel_index(i, (12, 3072)) for i in np.argsort(scores.flatten())][::-1][:topk]
    s0 = [scores[n] for n in n0]

    n1 = [np.unravel_index(i, (12, 3072)) for i in np.argsort(scores1.flatten())][::-1][:topk]
    s1 = [scores1[n] for n in n1]

    d0 = {n: s for n, s in zip(n0, s0)}
    d1 = {n: s for n, s in zip(n1, s1)}

    
    return d0, d1, n0, n1


def get_top_neurons(topk, exp_name, dataset_name, rs):
    with open(f'../data/scores_0_{dataset_name}_{exp_name}_rs_{rs}_cls.pkl', 'rb') as f:
        scores_alltopics = pickle.load(f)
    with open(f'../data/scores_1_{dataset_name}_{exp_name}_rs_{rs}_cls.pkl', 'rb') as f:
        scores1_alltopics = pickle.load(f)

    scores = np.concatenate([s for topic, s in scores_alltopics.items()])
    scores1 = np.concatenate([s for topic, s in scores1_alltopics.items()])
    scores = np.mean(np.max(scores, axis=2), axis=0)
    scores1 = np.mean(np.max(scores1, axis=2), axis=0)

    with open(f'../data/scores_detection_0_{dataset_name}_{exp_name}_rs_{rs}_cls.pkl', 'rb') as f:
        scores_det_alltopics = pickle.load(f)
    with open(f'../data/scores_detection_1_{dataset_name}_{exp_name}_rs_{rs}_cls.pkl', 'rb') as f:
        scores1_det_alltopics = pickle.load(f)

    scores_det = np.concatenate([s for topic, s in scores_det_alltopics.items()])
    scores1_det = np.concatenate([s for topic, s in scores1_det_alltopics.items()])
    scores_det = np.max(np.max(scores_det, axis=2), axis=0)
    scores1_det = np.max(np.max(scores1_det, axis=2), axis=0)

    d0, d1, n0, n1 = get_ranking_dict(scores, scores1, topk)
    d0det, d1det, n0det, n1det = get_ranking_dict(scores_det, scores1_det, topk)

    # inters = set(n0).union(set(n1)) - set(n0det).union(set(n1det))
    inters = set(n0).intersection(set(n1))
    # s = [[i, np.max([d0[i], d1[i]])] for i in inters]
    s = []
    for i in inters:
        if i in d0.keys() and i in d1.keys():
            s.append([i, np.max([d0[i], d1[i]])])
        elif i in d0.keys():
            s.append([i, d0[i]])
        else:
            s.append([i, d1[i]])
    s = sorted(s, key=lambda x: x[1], reverse=True)
    
    return n0, n1, s