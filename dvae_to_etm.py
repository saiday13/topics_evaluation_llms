import os
import json
import pickle
import random
import numpy as np
from scipy import sparse
from pathlib import Path
from scipy.io import savemat, loadmat

infile_path = './data/wikitext/processed_wiki/'
outfile_path = './data/wikitext/processed_wiki/etm-format/'
wiki_train_dtm = sparse.load_npz(Path(infile_path, "train.dtm.npz"))
wiki_val_dtm = sparse.load_npz(Path(infile_path, "val.dtm.npz"))
wiki_ts_dtm = sparse.load_npz(Path(infile_path, "ts.dtm.npz"))

wiki_full_dtm = sparse.load_npz(Path(infile_path, "full.dtm.npz"))
wiki_full_filtered_dtm = sparse.load_npz(Path(infile_path, "full_filtered.dtm.npz"))
wiki_tr_va_filtered_dtm = sparse.load_npz(Path(infile_path, "full_train_val_filtered.dtm.npz"))

def create_val_dtm():
    # prepate val.dtm.npz file
    wiki_val_size = 4200
    rand_ind = np.random.randint(0, wiki_full_filtered_dtm.shape[0], wiki_val_size)
    wiki_val = wiki_full_filtered_dtm[rand_ind, :]
    # save val set
    print("saving val set...")
    sparse.save_npz(Path(infile_path, "val.dtm.npz"), wiki_val)

def create_ts_dtm():
    # prepate val.dtm.npz file
    wiki_ts_size = 4200
    rand_ind = np.random.randint(0, wiki_tr_va_filtered_dtm.shape[0], wiki_ts_size)
    wiki_ts = wiki_tr_va_filtered_dtm[rand_ind, :]
    # save val set
    print("saving val set...")
    sparse.save_npz(Path(infile_path, "ts.dtm.npz"), wiki_ts)

def json_to_pkl():
    # Converting json file into pickle file
    print('converting json file to pickle file...')
    # read json file
    with open(infile_path + 'vocab.json', 'r') as infile:
        json_obj = json.load(infile)
    pkl_obj = pickle.loads(pickle.dumps(json_obj))
    vocab = list(pkl_obj.keys())

    # write into pkl file
    if not os.path.isdir(outfile_path):
        os.system('mkdir -p ' + outfile_path)
    with open(outfile_path + 'vocab.pkl', 'wb') as outfile:
        pickle.dump(vocab, outfile)

    return vocab

# create val/ts dtm files
#create_val_dtm()
#create_ts_dtm()

# print(wiki_full_filtered_dtm.shape)
# print(wiki_train_dtm.shape)
vocab = json_to_pkl()
n_docs_tr = wiki_train_dtm.shape[0]
n_docs_val = wiki_val_dtm.shape[0]
n_docs_ts = wiki_ts_dtm.shape[0]
n_docs_full = wiki_full_dtm.shape[0]

# Split test set in 2 halves
print('splitting test documents in 2 halves...')
ts_h1 = np.arange(0, n_docs_ts/2, dtype=int)
ts_h2 = np.arange(n_docs_ts/2, n_docs_ts, dtype=int)
bow_ts_h1 = wiki_ts_dtm[ts_h1]
bow_ts_h2 = wiki_ts_dtm[ts_h2]

n_docs_ts_h1 = bow_ts_h1.shape[0]
n_docs_ts_h2 = bow_ts_h2.shape[0]

# split bow intro token/value pairs
print('splitting bow into token/value pairs and saving to disk...')
def split_bow(bow_in, n_docs):
    indices = [[w for w in bow_in[doc,:].indices] for doc in range(n_docs)]
    counts = [[c for c in bow_in[doc,:].data] for doc in range(n_docs)]
    return indices, counts

# split train
bow_tr_tokens, bow_tr_counts = split_bow(wiki_train_dtm, n_docs_tr)
savemat(outfile_path + 'bow_tr_tokens.mat', {'tokens': bow_tr_tokens}, do_compression=True)
savemat(outfile_path + 'bow_tr_counts.mat', {'counts': bow_tr_counts}, do_compression=True)
#del vocab
del bow_tr_tokens
del bow_tr_counts
del n_docs_tr

# split val
bow_val_tokens, bow_val_counts = split_bow(wiki_val_dtm, n_docs_val)
savemat(outfile_path + 'bow_va_tokens.mat', {'tokens': bow_val_tokens}, do_compression=True)
savemat(outfile_path + 'bow_va_counts.mat', {'counts': bow_val_counts}, do_compression=True)
del bow_val_tokens
del bow_val_counts
del n_docs_val

# split test
bow_ts_tokens, bow_ts_counts = split_bow(wiki_ts_dtm, n_docs_ts)
savemat(outfile_path + 'bow_ts_tokens.mat', {'tokens': bow_ts_tokens}, do_compression=True)
savemat(outfile_path + 'bow_ts_counts.mat', {'counts': bow_ts_counts}, do_compression=True)
del bow_ts_tokens
del bow_ts_counts
del n_docs_ts

bow_ts_h1_tokens, bow_ts_h1_counts = split_bow(bow_ts_h1, n_docs_ts_h1)
savemat(outfile_path + 'bow_ts_h1_tokens.mat', {'tokens': bow_ts_h1_tokens}, do_compression=True)
savemat(outfile_path + 'bow_ts_h1_counts.mat', {'counts': bow_ts_h1_counts}, do_compression=True)
del bow_ts_h1
del bow_ts_h1_tokens
del bow_ts_h1_counts

bow_ts_h2_tokens, bow_ts_h2_counts = split_bow(bow_ts_h2, n_docs_ts_h2)
savemat(outfile_path + 'bow_ts_h2_tokens.mat', {'tokens': bow_ts_h2_tokens}, do_compression=True)
savemat(outfile_path + 'bow_ts_h2_counts.mat', {'counts': bow_ts_h2_counts}, do_compression=True)
del bow_ts_h2
del bow_ts_h2_tokens
del bow_ts_h2_counts

print('Data ready !!')
print('*************')