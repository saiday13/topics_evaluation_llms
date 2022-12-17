import re
import csv
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr, pearsonr

dataset = ['20ng', 'wiki']
prompt_intr = ['p1_v6', 'p2_v6', 'p3_v6', 'p4_v6', 'p5_v6', 'p6_v6']
version_rating = ['v1','v2','v3','v4']

def load_gpt3_resp(file_path):
    reader = open(file_path)
    data = [line.replace("\n", "") for line in reader.readlines()]
    counts = re.findall(r'\[.*?\]', data[0])
    counts = counts[0].replace('[','').replace(']','').replace(',','').split()
    counts = [int(num) for num in counts]

    return counts


def load_npmis(model, data):
    with open('./results/'+model+'-'+data+'/npmis.txt') as file:
        npmis = file.readlines()
    npmis = [float(term.replace('\n', '')) for term in npmis]
    return npmis

def collect_metrics():
    model = ['mallet', 'dvae', 'etm']
    header = ['dataset', 'model', 'npmi']
    gpt3_path = './transformer-tests/'

    with open(Path(gpt3_path, 'intrusion_metrics.csv'), 'w', encoding='UTF8') as file:
        header_intr = header + prompt_intr
        writer = csv.writer(file)
        writer.writerow(header_intr)
        for d in dataset:
            for m in model:

            # for j in range(len(prompt)):
                # gpt3_intr_mallet = load_gpt3_resp(Path(model[0], d, 'intr_' + prompt[j] + '_counts.txt'))
                # gpt3_intr_dvae = load_gpt3_resp(Path(model[1], d, 'intr_' + prompt[j] + '_counts.txt'))
                # gpt3_intr_etm = load_gpt3_resp(Path(model[2], d, 'intr_' + prompt[j] + '_counts.txt'))
                # for i in range(len(gpt3_intr_mallet)):
                #     row = [d, prompt[j], gpt3_intr_mallet[i], gpt3_intr_dvae[i], gpt3_intr_etm[i]]

                npmi = load_npmis(m, d)
                gpt3_intr = [load_gpt3_resp(Path(gpt3_path, m, d, 'intr_' + p + '_counts.txt')) for p in prompt_intr]
                for i in range(len(gpt3_intr[0])):
                    row = [d, m, npmi[i], gpt3_intr[0][i], gpt3_intr[1][i],
                           gpt3_intr[2][i], gpt3_intr[3][i], gpt3_intr[4][i], gpt3_intr[5][i]]
                    writer.writerow(row)


    with open(Path(gpt3_path, 'rating_metrics.csv'), 'w', encoding='UTF8') as file:
        header_rat = header + version_rating
        writer = csv.writer(file)
        writer.writerow(header_rat)
        for d in dataset:
            for m in model:
                npmi = load_npmis(m, d)
                gpt3_rating = [load_gpt3_resp(Path(gpt3_path, m, d, 'rating_p3_' + v + '_counts.txt')) for v in version_rating]
                for i in range(len(gpt3_rating[0])):
                    row = [d, m, npmi[i], gpt3_rating[0][i], gpt3_rating[1][i],
                           gpt3_rating[2][i], gpt3_rating[3][i]]
                    writer.writerow(row)



collect_metrics()
dfs_intr = pd.read_csv('./transformer-tests/intrusion_metrics.csv')

with open('./transformer-tests/metrics.txt', 'w') as f:
    for d in dataset:
        data_intr = dfs_intr[dfs_intr['dataset'] == d]
        npmi_intr = data_intr['npmi']
        for p in prompt_intr:
            p_val = data_intr[p]
            acc = np.mean(p_val)
            variance = np.var(p_val)
            spear_rho, spear_p = spearmanr(npmi_intr, p_val)
            pear_rho, pear_p = pearsonr(npmi_intr, p_val)
            metrics = {
                "task":'intrusion',
                "dataset": d,
                "prompt": p,
                "mean": acc,
                "var": variance,
                "spear_rho": spear_rho,
                "spear_p": spear_p,
                "pear_rho": pear_rho,
                "pear_p": pear_p
            }
            f.write(str(metrics) + '\n')

            print(metrics)



dfs_rating = pd.read_csv('./transformer-tests/rating_metrics.csv')

with open('./transformer-tests/metrics.txt', 'a') as f:
    for d in dataset:
        data_rating = dfs_rating[dfs_rating['dataset'] == d]
        npmi_rating = data_rating['npmi']
        for v in version_rating:
            p_val = data_rating[v]
            acc = np.mean(p_val)
            variance = np.var(p_val)
            spear_rho, spear_p = spearmanr(npmi_rating, p_val)
            pear_rho, pear_p = pearsonr(npmi_rating, p_val)
            metrics = {
                "task":'rating',
                "dataset": d,
                "version": v,
                "mean": acc,
                "var": variance,
                "spear_rho": spear_rho,
                "spear_p": spear_p,
                "pear_rho": pear_rho,
                "pear_p": pear_p
            }
            f.write(str(metrics) + '\n')

            print(metrics)