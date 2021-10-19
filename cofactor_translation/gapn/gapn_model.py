import os
import sys
from tqdm import tqdm
import pandas as pd
import numpy as np
from scipy.stats import linregress
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
# Kernel ridge regression (KRR) combines ridge regression (linear least squares with l2-norm regularization) with the kernel trick.
# https://scikit-learn.org/stable/modules/generated/sklearn.kernel_ridge.KernelRidge.html
from sklearn.kernel_ridge import KernelRidge

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(context='notebook', style='ticks',
        color_codes=True, rc={'legend.frameon': False})

import torch
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")


alg = 'KernelRidge'
kernel = 'laplacian'

gapn_data_file = "20210831-GapN-data-up-to-SM130.csv"
df = pd.read_csv(gapn_data_file, index_col=0)
df = df.dropna()
# drop the barcode of the sequences: MGSSHHHHHHSSGLVPRGSH
df['Sequence'] = df['Sequence'].apply(lambda x: x.replace('MGSSHHHHHHSSGLVPRGSH', ''))
print(df.head(2))

#model_name = "esm1_t6_43M_UR50S"
model_name = "esm1b_t33_650M_UR50S"
#torch.hub.set_dir('/tmp/.cache/torch')
torch.hub.set_dir('/gpfs/alpine/scratch/jlaw/bie108/torch/2021-10-hub/')
# TODO Try this model: esm1v_t33_650M_UR90S
model, alphabet = torch.hub.load("facebookresearch/esm", model_name)

batch_converter = alphabet.get_batch_converter()
batch_labels, batch_strs, batch_tokens = batch_converter(list(df['Sequence'].items()))

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
print("Generating embeddings of {len(df)} sequences using batch size=8")
batch_dataloader = torch.utils.data.DataLoader(batch_tokens, batch_size=8)

output = []
with torch.no_grad():
    for batch in tqdm(batch_dataloader):
        result = model(batch.to(device), repr_layers=[33])  # because this is the 6-layer transformer
        output += [result['representations'][33].detach().cpu().numpy()]

outputs = np.vstack(output)

# Generate per-sequence representations via averaging
# NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
representations = []
for output, seq in zip(outputs, df['Sequence']):
    representations += [output[1 : len(seq) + 1].mean(0)]
    
representations = np.vstack(representations)
representations.shape


# now run K-fold CV using ridge regression
X = representations
y = df['NAD+_initial_rate'].astype(float)

kf = KFold(n_splits=5, shuffle=True)
print(kf)

#alphas = [0.0001, 0.001, 0.01, 0.1, .5, 1, 2, 10, 100]
alphas = [1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1, 1, 10]
# alphas = [.01]
repetitions = 10
best_alpha = 0
best_rmse = 10000
best_y_pred = []
print("model_name\talpha\tavg_slope\tavg_mae\tavg_rmse")
for alpha in alphas:
    y_preds = []
    maes = []
    rmses = []
    slopes = []
    for rep in range(repetitions):
        y_pred = np.zeros(len(y))
        for train_index, test_index in kf.split(X):
        #     print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            if alg.lower() == "kernelridge":
                clf = KernelRidge(alpha=alpha, kernel=kernel)
            elif alg.lower() == "ridge":
                clf = Ridge(alpha=alpha)
            clf.fit(X_train, y_train)
            y_pred[test_index] = clf.predict(X_test)

        y_preds.append(y_pred)
        # mean avg error
        mae = np.sum(np.abs(y - y_pred))
        # also compute the root mean squared error
        rmse = np.sqrt(np.sum((y - y_pred)**2) / len(y))
        maes.append(mae)
        rmses.append(rmse)
        # and the slope
        slope, intercept, r_value, p_value, std_err = linregress(y, y_pred)
        slopes.append(slope)
        
    avg_mae = np.mean(mae)
    avg_slope = np.mean(slopes)
    avg_rmse = np.mean(rmses)
    print(f"{model_name}\t{alpha}\t{avg_slope:0.2f}\t{avg_mae:0.2f}\t{avg_rmse:0.2f}")
    
    if avg_rmse < best_rmse:
        best_rmse = avg_rmse
        best_alpha = alpha
        best_y_pred = y_preds

    # now plot
    df2 = pd.DataFrame({'y': y, 'y_pred': np.mean(y_preds, axis=0)})
    print(df2.head(2))

    f, ax = plt.subplots()
    color = "#3291a8"
    std_dev = np.std(y_preds, axis=0)
    ax.errorbar(df2['y'], df2['y_pred'], yerr=np.max(y_preds, axis=0) - df2['y_pred'], fmt='o', lolims=True, ms=0, lw=1, ls=None, capsize=.5, ecolor=color)
    ax.errorbar(df2['y'], df2['y_pred'], yerr=df2['y_pred'] - np.min(y_preds, axis=0), fmt='o', uplims=True, ms=0, lw=1, ls=None, capsize=.5, ecolor=color)
    ax.errorbar(df2['y'], df2['y_pred'], yerr=std_dev, fmt="o", ms=8, lw=3, color=color)
    # df2.plot.scatter(x='y', y="y_pred", s=1, ax=ax)
    plt.suptitle(f"GapN NAD+ {alg}-{kernel} (alpha={alpha})", fontsize=14)
    ax.set_title(f"5-fold CV ({repetitions}x), ESM model: {model_name}", fontsize=12)
    ax.set_ylabel("y_pred")
    ax.set_xlabel("y")

    out_dir = "./viz/vary-alpha/"
    out_file = f"{out_dir}/gapn-nadh-{alg}-{kernel}-{model_name}-a{str(alpha).replace('.','_')}.svg"
    print(out_file)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    plt.savefig(out_file, bbox_inches='tight')
    plt.savefig(out_file.replace('.svg','.png'), bbox_inches='tight')
    plt.savefig(out_file.replace('.svg','.pdf'), bbox_inches='tight')


y_preds = best_y_pred
print(f"best_rmse: {best_rmse:0.2f}, best_alpha: {best_alpha}")
