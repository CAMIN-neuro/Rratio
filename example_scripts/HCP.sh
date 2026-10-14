#!/bin/bash

config=../config/HCP.ini

# training
python -W ignore ../code/Rratio_pFIC_training_HCP.py $config
python -W ignore ../code/Rratio_pFIC_transpose_matrix.py $config

# validation
python -W ignore ../code/Rratio_pFIC_validation.py $config

# test
python -W ignore ../code/Rratio_pFIC_test.py $config
