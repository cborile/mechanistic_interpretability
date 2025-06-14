import argparse
import os
from tqdm import tqdm
import torch
import gc
import torch
import numpy as np
import pickle
import pandas as pd

from inspector_cls import load_model
from utils import seed_everything


def get_ig_scores(st_pairs, target_label, n_prompts_max, model_path, bs=1, ig_steps=5):
    all_scores = {}
    all_scores_bsl = {}
    all_scores_detection = {}
    all_model_labels = {}
    for topic, st in st_pairs.items():
        print(topic)
        all_scores[topic] = []
        all_scores_bsl[topic] = []
        all_scores_detection[topic] = []
        model_labels = []
        c = 0
        with tqdm(total=n_prompts_max-1, desc="Getting attribution scores for each text...", leave=True, position=0) as pbar:
            for text, mask_indices, mask_indices_bsl in st[target_label]:
                if len(mask_indices)>=5:  # temporary magic fix, guarantees same dimensions
                    c+=1
                    inspector = load_model(model_path)
                    # this is because the model prediction could be different from the real label
                    # encoded_input = inspector._prepare_inputs(text)
                    # outputs = inspector.model(**encoded_input)
                    # label = outputs.logits.softmax(1).argmax().cpu().detach().item()
                    # model_labels.append(label)
                    label = target_label
                    scores = inspector.get_scores(text, label, mask_indices[:5], batch_size=bs, ig_steps=ig_steps)
                    # scores_bsl = inspector.get_scores(text, label, mask_indices_bsl[:5], batch_size=bs, ig_steps=ig_steps)
                    scores_detection = inspector.get_scores(text, label, [0], batch_size=bs, ig_steps=ig_steps)
        
                    all_scores[topic].append(scores.cpu().detach().numpy())
                    # all_scores_bsl[topic].append(scores_bsl.cpu().detach().numpy())
                    all_scores_detection[topic].append(scores_detection.cpu().detach().numpy())
                    inspector.model.zero_grad()
                    del inspector
                    gc.collect()
                    torch.cuda.empty_cache()
                    pbar.update(1)
                if c>=n_prompts_max:
                    break
        all_model_labels[topic] = model_labels
        all_scores[topic] = np.array(all_scores[topic])
        # all_scores_bsl[topic] = np.array(all_scores_bsl[topic])
        all_scores_detection[topic] = np.array(all_scores_detection[topic])
    return all_scores, all_scores_detection, all_model_labels # all_scores_bsl, 


def main():
    parser = argparse.ArgumentParser(description='Text detector (BERT-based)')
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--keywords', type=int, default=5) 
    parser.add_argument('--texts', type=int, default=50)
    parser.add_argument('--data_path', type=str, default="../data/")
    parser.add_argument('--output_path', type=str, default="../outputs/")
    parser.add_argument('--model_path', type=str, default="../models/")
    parser.add_argument('--dataset', type=str, default="DAIGTV2")
    parser.add_argument('--base_model', type=str, default="bert-base-cased")
    parser.add_argument('--rs', type=int, default=0)
    parser.add_argument('--max_tokens', type=int, default=512)
    args = parser.parse_args()
    print(args)

    seed_everything(args.rs)

    input_path = os.path.join(args.data_path, f"st_pairs_{args.dataset}_{args.exp_name}_rs_{args.rs}.pkl")
    with open(input_path, 'rb') as f:
        st_pairs = pickle.load(f)

    print("topics considered:", st_pairs.keys())

    for label in [0, 1]:
        print(f"scores for label {label}")
        
        scores, scores_detection, _ = get_ig_scores(
            st_pairs, 
            label, 
            args.texts, 
            f'../models/detector-{args.base_model}-{args.dataset}-{args.exp_name}'
        )

        print("save scores")
        output_path = os.path.join(args.data_path, f"scores_{label}_{args.dataset}_{args.exp_name}_rs_{args.rs}_cls.pkl")
        with open(output_path, 'wb') as f:
            pickle.dump(scores, f)

        # print("save scores bsl")
        # output_path = os.path.join(args.data_path, f"scores_bsl_{label}_{args.dataset}_{args.exp_name}_rs_{args.rs}_cls.pkl")
        # with open(output_path, 'wb') as f:
        #     pickle.dump(scores_bsl, f)

        print("save scores detection")
        output_path = os.path.join(args.data_path, f"scores_detection_{label}_{args.dataset}_{args.exp_name}_rs_{args.rs}_cls.pkl")
        with open(output_path, 'wb') as f:
            pickle.dump(scores_detection, f)

        # # ranking for "topic"
        # print("ranking for scores")
        # data = []
        # for topic, score_list in scores.items():
        #     print(topic)
        #     for i in range(score_list.shape[0]):  # for in texts
        #         for j in  range(score_list.shape[2]): # for in keywords
        #             # print(f"text {i} keyword {j}")
        #             top_neurons = np.argsort(np.abs(score_list[i, :, j, :]).flatten())[::-1]
        #             for c, k in enumerate(top_neurons):
        #                 layer, neuron = np.unravel_index(k, score_list[i, :, j, :].shape)
        #                 data.append([topic, i, j, layer, neuron, score_list[i, layer, j, neuron], c])
    
        # df = pd.DataFrame(data, columns=['topic', 'text', 'keyword', 'layer', 'neuron', 'score', 'ranking'])
        # print(df.shape)
        # output_path = os.path.join(args.output_path, f"neuron_ranking_{label}_{args.dataset}_{args.exp_name}_rs_{args.rs}_cls.csv")
        # df.to_csv(output_path)

        # # ranking for "topic" baseline
        # print("ranking for scores")
        # data_bsl = []
        # for topic, score_list in scores_bsl.items():
        #     print(topic)
        #     for i in range(score_list.shape[0]):  # for in texts
        #         for j in  range(score_list.shape[2]): # for in keywords
        #             # print(f"text {i} keyword {j}")
        #             top_neurons = np.argsort(np.abs(score_list[i, :, j, :]).flatten())[::-1]
        #             for c, k in enumerate(top_neurons):
        #                 layer, neuron = np.unravel_index(k, score_list[i, :, j, :].shape)
        #                 data_bsl.append([topic, i, j, layer, neuron, score_list[i, layer, j, neuron], c])
    
        # df_bsl = pd.DataFrame(data_bsl, columns=['topic', 'text', 'keyword', 'layer', 'neuron', 'score', 'ranking'])
        # print(df_bsl.shape)
        # output_path = os.path.join(args.output_path, f"neuron_ranking_bsl_{label}_{args.dataset}_{args.exp_name}_rs_{args.rs}_cls.csv")
        # df_bsl.to_csv(output_path)

        # # ranking for detection
        # print("rankings for detection")
        # data_detection = []
        # for topic, score_list in scores_detection.items():
        #     print(topic)
        #     for i in range(score_list.shape[0]):  # for in texts
        #         for j in  range(score_list.shape[2]): # for in keywords
        #             # print(f"text {i} keyword {j}")
        #             top_neurons = np.argsort(np.abs(score_list[i, :, j, :]).flatten())[::-1]
        #             for c, k in enumerate(top_neurons):
        #                 layer, neuron = np.unravel_index(k, score_list[i, :, j, :].shape)
        #                 data_detection.append([topic, i, j, layer, neuron, score_list[i, layer, j, neuron], c])

        # df_det = pd.DataFrame(data_detection, columns=['topic', 'text', 'keyword', 'layer', 'neuron', 'score', 'ranking'])
        # print(df_det.shape)
        # output_path = os.path.join(args.output_path, f"neuron_ranking_detection_{label}_{args.dataset}_{args.exp_name}_rs_{args.rs}_cls.csv")
        # df_det.to_csv(output_path)

    
if __name__ == "__main__":
    main()