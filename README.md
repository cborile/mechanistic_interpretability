# Mechanistic Interpretability

Mechanistic interventions in BERT transformer-based neural models for concept extraction 

## To run the code:
### 1. Training the model:
```
python train_detection_model.py --topics_list <topic1 topic2 ...> --exp_name <experiment_name> --topic_col_name <name> --dataset=<dataset>
```

### 2. Keywords Extraction:
```
python extract_keywords.py --exp_name <experiment_name> --topic_col_name <name> --dataset=<dataset>
```

### 3. Compute Scores:
```
python get_scores_cls.py --exp_name <experiment_name> --dataset=<dataset>
```

Supported datasets for training are currently `DAIGTV2`, `HC3`, `DAIGTV2lda`.

## Check Results:
The Jupyter notebook `validation.ipynb` allows to run the validation of the computed confounding neurons scores.