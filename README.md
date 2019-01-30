# AAAI-19 paper: "Neural Relation Extraction Within and Across Sentence Boundaries"

This repository consists of the following data

1. data:
    The folder contains MUC6 entity and relation annotations.

2. code
    Contains the codebase for idepnn models

    - hyper_param_search.py - specify various hyper parameter configurations in this file

    Features:
        Major features used for training are:
        - position_indicator_embedding
        - context window
        - postag
        - entity_class
        - subtree

    Training:
        THEANO_FLAGS='float=float64' python hyper_param_search.py

    Prediction:
        Set "train_model" = False, "reload_model" = True and load model by specifying the path in "reload_path"


    ***** Detailed documentation in directory: "./documentation" *****





