#!/bin/bash

config=../../config/ADNI_acm/ADNI_1.ini

# training
python -W ignore ../../code/Rratio_pFIC_training_others.py $config
python -W ignore ../../code/Rratio_pFIC_transpose_matrix.py $config

# validation
python -W ignore ../../code/Rratio_pFIC_validation.py $config

# test
python -W ignore ../../code/Rratio_pFIC_test.py $config
