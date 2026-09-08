# /usr/bin/env python
"""
Written by Shaoshi Zhang and CBIG under MIT license:
https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md
"""

import os
import shutil
import sys
import warnings
import configparser
import torch
import numpy as np
import scipy.io as sio
import Rratio_pFIC_misc as misc


def CBIG_mfm_test_main(parser):
    input_path = parser['validation']['output_path']
    output_path = parser['test']['output_path']
    FCD = parser['test']['FCD_test']
    SC = parser['test']['SC_test']
    FC = parser['test']['FC_test']
    rE_min = float(parser['system']['rE_min'])
    rE_max = float(parser['system']['rE_max'])
    FC_corr_weight = float(parser['training']['FC_corr_weight'])
    FC_L1_weight = float(parser['training']['FC_L1_weight'])
    FCD_weight = float(parser['training']['FCD_KS_weight'])
    gpu_number = int(parser['system']['GPU_index'])
    if gpu_number != -1:
        torch.cuda.set_device(gpu_number)

    if not os.path.exists(output_path):
        os.makedirs(output_path)
        os.makedirs(output_path + 'simulation/')
    else:
        shutil.rmtree(output_path)
        os.makedirs(output_path)
        os.makedirs(output_path + 'simulation/')

    torch.cuda.manual_seed(1)

    n_ave = int(parser['test']['n_ave'])
    n_dup = int(parser['test']['n_dup'])
    n_set = int(n_dup / n_ave)

    SC_data = np.load(SC)
    n_node = SC_data.shape[0]
    
    # process_id + index in process + train_loss + validation_loss + parameters
    num_process = len(eval(parser['system']['random_seed']))
    vali_raw_all = np.zeros((4 * n_node + 10, 1)) 
    for i in range(1, num_process + 1):
        load_file = parser['validation']['output_file'] + '_' + str(i) + '.csv'
        load_path = os.path.join(input_path, load_file)
        if not os.path.exists(load_path):
            continue

        xmin = misc.csv_matrix_read(load_path)
        index_mat = np.zeros((2, xmin.shape[1])) # (2, 100)
        index_mat[0, :] = i
        index_mat[1, :] = np.arange(xmin.shape[1])
        xmin = np.concatenate((index_mat, xmin), axis=0)

        vali_raw_all = np.concatenate((vali_raw_all, xmin), axis=1)

    vali_raw_all = vali_raw_all[:, 1:] # delete dummy value
    vali_index = np.argsort(vali_raw_all[9, :])  # sort the total cost(validation)
    vali_sort_all = vali_raw_all[:, vali_index]

    n_solution = int(parser['test']['n_solution']); i = 0
    similarity_threshold = float(parser['test']['similarity_threshold'])
    vali_sel = np.zeros((vali_raw_all.shape[0], n_solution)) # selected validation
    p_set = np.zeros(n_solution); p = 0
    # i = solution number and p = parameter index
    while i < n_solution and p < vali_raw_all.shape[1]:
        # results of similarity threshold
        corr_t = np.zeros(n_solution, dtype=bool)
        
        # each params correlation between selected and new
        corr_tr = np.zeros((n_solution, 4))  

        for j in range(n_solution):
            wEE_intra_corr = np.corrcoef(vali_sel[10:10 + n_node, j:j + 1].T,
                                   vali_sort_all[10:10 + n_node, p:p + 1].T)
            wEI_corr = np.corrcoef(vali_sel[10 + n_node:10 + 2 * n_node, j:j + 1].T,
                            vali_sort_all[10 + n_node:10 + 2 * n_node, p:p + 1].T)
            sigma_corr = np.corrcoef(vali_sel[10 + 2 * n_node:10 + 3 * n_node, j:j + 1].T,
                    vali_sort_all[10 + 2 * n_node:10 + 3 * n_node, p:p + 1].T)
            wEE_inter_corr = np.corrcoef(vali_sel[10 + 3 * n_node:10 + 4 * n_node, j:j + 1].T,
                            vali_sort_all[10 + 3 * n_node:10 + 4 * n_node, p:p + 1].T)

            corr_tr[j, 0] = wEE_intra_corr[0, 1]
            corr_tr[j, 1] = wEI_corr[0, 1]
            corr_tr[j, 2] = sigma_corr[0, 1]
            corr_tr[j, 3] = wEE_inter_corr[0, 1]

        for k in range(n_solution):
            corr_t[k] = (np.abs(corr_tr[k, :] > similarity_threshold)).all()
        
        if not corr_t.any():
            vali_sel[:, i] = vali_sort_all[:, p]
            p_set[i] = p
            i += 1
        p += 1

    # result_all = id + num + train_loss + val_loss + test_loss + parameters
    result_save = np.zeros((4 * n_node + 14, n_solution))
    result_save[0:10, :] = vali_sel[0:10, :]
    result_save[14:, :] = vali_sel[10:, :]

    # simulation with selected 10 parameters
    for j in range(n_solution):
        print('[Solution ' + str(j + 1) + '] Test starts ...')
        test_cost = np.zeros((4, n_set))

        arx = np.tile(vali_sel[10:, j:j + 1], [1, n_set]) # repetation n_set time in axis=1
        total_cost, corr_cost, L1_cost, fcd_cost, bold, S_E, S_I, r_E, w_IE = \
            misc.CBIG_combined_cost_test(arx, n_ave, SC, FC, FCD, rE_min, rE_max, [FC_corr_weight,
                                                  FC_L1_weight], FCD_weight, parser)
        test_cost[0, :] = corr_cost
        test_cost[1, :] = L1_cost
        test_cost[2, :] = fcd_cost
        test_cost[3, :] = total_cost
        
        test_file = os.path.join(output_path, 'solution_' + str(j + 1) + '.csv')
        with open(test_file, 'a'):
            np.savetxt(test_file, test_cost, delimiter=',')

        result_save[10, j] = np.nanmean(test_cost[0, :])
        result_save[11, j] = np.nanmean(test_cost[1, :])
        result_save[12, j] = np.nanmean(test_cost[2, :])
        result_save[13, j] = np.nanmean(test_cost[3, :])

        bold = bold.cpu().numpy()
        BOLD = {'TC': bold}
        S_E_save = S_E.cpu().numpy()
        S_E = {'S_E': S_E_save}
        S_I_save = S_I.cpu().numpy()
        S_I = {'S_I': S_I_save}
        r_E_save = r_E.cpu().numpy()
        r_E = {'r_E': r_E_save}

        sio.savemat(output_path + 'simulation/TC_' + str(j + 1) + '.mat', BOLD)
        sio.savemat(output_path + 'simulation/S_E_' + str(j + 1) + '.mat', S_E)
        sio.savemat(output_path + 'simulation/S_I_' + str(j + 1) + '.mat', S_I)
        sio.savemat(output_path + 'simulation/r_E_' + str(j + 1) + '.mat', r_E)
        print('*******************************************')

        test_file_all = os.path.join(output_path, 'test_all.csv')
        with open(test_file_all, 'a'):
            np.savetxt(test_file_all, result_save, delimiter=',')


if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    config = sys.argv[1]
    parser = configparser.ConfigParser()
    parser.read(config)
    CBIG_mfm_test_main(parser)
