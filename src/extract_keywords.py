import argparse
import os
from tqdm import tqdm
import torch
#from transformer_lens import HookedEncoder
# import transformer_lens.utils as utils
from transformers import AutoTokenizer
torch.set_grad_enabled(False)

import pandas as pd


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import torch
import numpy as np
from transformers import AutoTokenizer
import pickle

from utils import seed_everything

def extract_top_words(data, topic_col, no_keywords):
    # Initialize the TfidfVectorizer
    vectorizer = TfidfVectorizer(stop_words='english')

    topic_top_tokens = {}
    topic_bottom_tokens = {}
    for topic in data[topic_col].unique():
        sub_corpus = data[data[topic_col] == topic]['text']
        tfidf_matrix = vectorizer.fit_transform(sub_corpus)
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
        top_tokens = tfidf_df.sum(axis=0).sort_values(ascending=False).head(no_keywords).index.tolist()
        topic_top_tokens[topic] = top_tokens

        bottom_tokens = tfidf_df.sum(axis=0).sort_values(ascending=False).tail(no_keywords).index.tolist()
        topic_bottom_tokens[topic] = bottom_tokens
    
    return topic_top_tokens, topic_bottom_tokens

def get_st_pairs(data, topic_top_tokens, topic_bottom_tokens, topics_list, tokenizer, topic_col, max_tokens=512):
    st_pairs = {}
    for topic in topics_list:
        sub_corpus = data[data[topic_col] == topic].copy()
        st_pairs_single_topic = {0: [], 1: []}
        for text, label in tqdm(sub_corpus[['text', 'label']].values):
            encoded_text = tokenizer(text, truncation=True, max_length=max_tokens)['input_ids']
            assert len(encoded_text)<=max_tokens
            indices_single_text = []
            indices_single_text_bsl = [] 
            for kw in topic_top_tokens[topic]:
                encoded_kw = tokenizer(kw, add_special_tokens=False)['input_ids']
                if len(encoded_kw)==1:
                    idxs = [i for i, tok in enumerate(encoded_text) if tok==encoded_kw[0]]
                    indices_single_text.extend(idxs)
                else:
                    continue
                    # print(kw, encoded_kw)
            indices_single_text = list(set(indices_single_text))

            for kw in topic_bottom_tokens[topic]:
                encoded_kw = tokenizer(kw, add_special_tokens=False)['input_ids']
                if len(encoded_kw)==1:
                    idxs = [i for i, tok in enumerate(encoded_text) if tok==encoded_kw[0]]
                    indices_single_text.extend(idxs)
                else:
                    continue
                    # print(kw, encoded_kw)
            indices_single_text_bsl = list(set(indices_single_text))
            
            if len(indices_single_text)>0:
                # indices_single_text_bsl = [i for i in range(np.max(indices_single_text)) if i not in indices_single_text]
                st_pairs_single_topic[label].append([text, indices_single_text, indices_single_text_bsl])
        st_pairs[topic] = st_pairs_single_topic
        print(topic, len(st_pairs[topic][0]), len(st_pairs[topic][1]))

    return st_pairs

def main():
    parser = argparse.ArgumentParser(description='Text detector (BERT-based)')
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--keywords', type=int, default=5) 
    parser.add_argument('--topic_col_name', type=str, default="prompt_name") 
    parser.add_argument('--data_path', type=str, default="../data/")
    parser.add_argument('--model_path', type=str, default="../models/")
    parser.add_argument('--dataset', type=str, default="DAIGTV2")
    parser.add_argument('--base_model', type=str, default="bert-base-cased")
    parser.add_argument('--rs', type=int, default=0)
    parser.add_argument('--max_tokens', type=int, default=512)
    args = parser.parse_args()
    print(args)

    seed_everything(args.rs)

    input_path = os.path.join(
        args.data_path, 
        "input", 
        f"data_train_{args.dataset}_{args.exp_name}_rs_{args.rs}.json"
        )
    train_data = pd.read_json(input_path, orient='records', lines=True)
    topics_list = train_data[args.topic_col_name].unique()
    print("topics considered:", topics_list)

    if args.dataset == "DAIGTV2lda":
        print("loading presaved top keywords")
        topkwds = pd.read_csv('../data/input/train_v2_drcat_02_lda_top_keywords.csv', index_col=0)
        topic_top_tokens = {k[0]: list(k[1:]) for k in topkwds.reset_index().values}
    elif args.dataset == "polarity":
        print("loading presaved top keywords")
        topkwds = pd.read_csv('../data/input/trainfakenews_ft_top_keywords.csv', index_col=0)
        topic_top_tokens = {k[0]: list(k[1:]) for k in topkwds.reset_index().values}
        topic_bottom_tokens = {k[0]: list(k[1:]) for k in topkwds.reset_index().values}
    else:
        topic_top_tokens, topic_bottom_tokens = extract_top_words(train_data, args.topic_col_name, args.keywords)

    # print(topic_bottom_tokens)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    st_pairs = get_st_pairs(train_data, topic_top_tokens, topic_bottom_tokens, topics_list, tokenizer, args.topic_col_name, max_tokens=args.max_tokens)
    output_path = os.path.join(args.data_path, f"st_pairs_{args.dataset}_{args.exp_name}_rs_{args.rs}.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(st_pairs, f)

if __name__ == "__main__":
    main()