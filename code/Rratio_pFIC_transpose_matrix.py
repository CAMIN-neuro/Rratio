# /usr/bin/env python
"""
Written by Shaoshi Zhang and CBIG under MIT license:
https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md
"""

import sys
from multiprocessing import Pool
from multiprocessing import set_start_method
import configparser
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def tranpose_matrix(parser, process_id):
    input_path = parser['training']['output_path']
    input_file = parser['training']['output_file']
    training_matrix = pd.read_csv(input_path + input_file + "_{}.csv".format(str(process_id)),
            header=None).T
    training_matrix.to_csv(input_path + input_file + "_{}.csv".format(str(process_id)), 
            index=False, header=False)

    FC_cost = np.array(training_matrix.iloc[-4, :])
    FC_L1_cost = np.array(training_matrix.iloc[-3, :])
    FCD_cost = np.array(training_matrix.iloc[-2, :])
    total_cost = np.array(training_matrix.iloc[-1, :])
    xdata = np.arange(len(FC_cost))
    plt.plot(xdata, FC_cost, label="FC corr cost")
    plt.plot(xdata, FC_L1_cost, label="FC L1 cost")
    plt.plot(xdata, FCD_cost, label="FCD cost")
    plt.plot(xdata, total_cost, label="Total cost")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.savefig(input_path + "learning_curve_" + str(process_id) + ".png")


if __name__ == "__main__":
    config = sys.argv[1]
    parser = configparser.ConfigParser()
    parser.read(config)
    num_thread = int(parser['system']['num_thread'])

    input_args = ()
    for i in range(num_thread):
        input_args_thread = [parser, i + 1]
        input_args = input_args + (input_args_thread, )

    set_start_method("spawn")
    p = Pool(num_thread)
    p.starmap(tranpose_matrix, input_args)
