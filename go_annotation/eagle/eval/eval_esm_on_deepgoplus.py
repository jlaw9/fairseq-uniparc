
import argparse
from itertools import islice
import os
import sys
import gzip
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from fairseq.data import data_utils
from fairseq.models.roberta import RobertaModel
from scipy.sparse import load_npz
from sklearn.metrics import precision_recall_curve
from tqdm import tqdm, trange
import pandas as pd
import pickle
from scipy.sparse import save_npz, load_npz
import math
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append('../../../')
#sys.path.insert(0, '..//fairseq_uniparc_fork')
from go_annotation.ontology import Ontology

sns.set(context='talk', style='ticks',
        color_codes=True, rc={'legend.frameon': False})

import fairseq
print(f"fairseq version: {fairseq.__version__}")
print(f"torch version: {torch.__version__}")
print(f"numpy version: {np.__version__}")

print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_esm_model(checkpoint_dir, data_dir, checkpoint):
    print(f"loading {checkpoint} from {checkpoint_dir}")
    x = fairseq.hub_utils.from_pretrained(
        checkpoint_dir,
        checkpoint_file=checkpoint,
        data_name_or_path=data_dir)
    print("\tdone")
    args, task, model = x["args"], x["task"], x["models"][0]
    model.task = task
    _ = model.eval()  # disable dropout (or leave in train mode to finetune)
    _ = model.to(device)
    
    return model


def normalize_logits(logits, _ancestor_array):
    bsz = logits.shape[0]
    index_tensor = logits.new_tensor(_ancestor_array, dtype=torch.int64)
    index_tensor = index_tensor.unsqueeze(0).expand((bsz, -1, -1))  # Array of ancestors, offset by one
    padded_logits = torch.nn.functional.pad(logits, (1, 0), value=float('inf'))  # Make 0 index return inf
    padded_logits = padded_logits.unsqueeze(-1).expand((-1, -1, index_tensor.shape[2]))
    normed_logits = torch.gather(padded_logits, 1, index_tensor)
    normed_logits, _ = torch.min(normed_logits, -1)

    return normed_logits


def encode(esm_model, sequence):
    sequence = '<s> ' + sequence.replace('B', 'D').replace('Z', 'E').replace('J', 'L')
#     max_positions = int(esm_model.max_positions())
    max_positions = 768
    encoded_sequence = esm_model.task.source_dictionary.encode_line(sequence, add_if_not_exist=False)[:max_positions]
    return encoded_sequence


def inputs_generator(roberta_model, filename, batch_size):
    # Update: change the raw files to gzipped files
    with gzip.open(filename, 'r') as f:
        encoded_lines = (encode(roberta_model, line.decode()) for line in f)
        for batch in iter(lambda: tuple(islice(encoded_lines, batch_size)), ()):
            yield data_utils.collate_tokens(
                batch, pad_idx=roberta_model.task.source_dictionary.pad()).long().to(device)


def make_predictions(esm_model, inputs_filename, _ancestor_array):

    # we can use a larger batch size for smaller models
    # but for the esm1b model, we run into GPU limitations pretty quickly
    batch_size = 1

    # quickly get the line count
    num_prots = 0
    with gzip.open(inputs_filename, 'r') as f:
        for line in f:
            num_prots += 1

    y_pred = []
    print(f"making predictions for proteins in {inputs_filename}")
    for inputs in tqdm(inputs_generator(esm_model, inputs_filename, batch_size), total=int(np.ceil(num_prots / batch_size))):

        assert inputs.shape[1] <= 1024
        assert inputs.max() <= 25

        with torch.no_grad():
            logits, _ = esm_model(inputs, features_only=True, classification_head_name='go_prediction')

            normed_logits = normalize_logits(logits, _ancestor_array)
            batch_pred = torch.sigmoid(normed_logits)
            y_pred += [batch_pred.detach().cpu().numpy()]
    
    y_pred = np.concatenate(y_pred)
    return y_pred


BIOLOGICAL_PROCESS = 'GO:0008150'
MOLECULAR_FUNCTION = 'GO:0003674'
CELLULAR_COMPONENT = 'GO:0005575'
FUNC_DICT = {
    'cc': CELLULAR_COMPONENT,
    'mf': MOLECULAR_FUNCTION,
    'bp': BIOLOGICAL_PROCESS}

NAMESPACES = {
    'cc': 'cellular_component',
    'mf': 'molecular_function',
    'bp': 'biological_process'
}

EXP_CODES = set([
    'EXP', 'IDA', 'IPI', 'IMP', 'IGI', 'IEP', 'TAS', 'IC',])
#    'HTP', 'HDA', 'HMP', 'HGI', 'HEP'])
CAFA_TARGETS = set([
    '10090', '223283', '273057', '559292', '85962',
    '10116',  '224308', '284812', '7227', '9606',
    '160488', '237561', '321314', '7955', '99287',
    '170187', '243232', '3702', '83333', '208963',
    '243273', '44689', '8355'])


def is_cafa_target(org):
    return org in CAFA_TARGETS


def is_exp_code(code):
    return code in EXP_CODES


def combine_with_diamond(
        preds, terms, go_rels, train_df, test_df,
        deepgoplus_dir, annotations):
    # the below is copeid from the UDSMProt function
    # to use the deepgoplus evaluation
    diamond_scores_file = f"{deepgoplus_dir}/test_diamond.res"
    # BLAST Similarity (Diamond)
    diamond_scores = {}
    with open(diamond_scores_file) as f:
        for line in f:
            it = line.strip().split()
            if it[0] not in diamond_scores:
                diamond_scores[it[0]] = {}
            diamond_scores[it[0]][it[1]] = float(it[2])

    prot_index = {}
    for i, row in enumerate(train_df.itertuples()):
        prot_index[row.proteins] = i

    blast_preds = []
    for i, row in enumerate(test_df.itertuples()):
        annots = {}
        prot_id = row.proteins
        # BlastKNN
        if prot_id in diamond_scores:
            sim_prots = diamond_scores[prot_id]
            allgos = set()
            total_score = 0.0
            for p_id, score in sim_prots.items():
                allgos |= annotations[prot_index[p_id]]
                total_score += score
            allgos = list(sorted(allgos))
            sim = np.zeros(len(allgos), dtype=np.float32)
            for j, go_id in enumerate(allgos):
                s = 0.0
                for p_id, score in sim_prots.items():
                    if go_id in annotations[prot_index[p_id]]:
                        s += score
                sim[j] = s / total_score
            ind = np.argsort(-sim)
            for go_id, score in zip(allgos, sim):
                annots[go_id] = score
        blast_preds.append(annots)

    preds_w_diamond = []
    alphas = {NAMESPACES['mf']: 0.55, NAMESPACES['bp']: 0.59, NAMESPACES['cc']: 0.46}
    for i, row in enumerate(test_df.itertuples()):
        annots_dict = blast_preds[i].copy()
        for go_id in annots_dict:
            annots_dict[go_id] *= alphas[go_rels.get_namespace(go_id)]
        for j, score in enumerate(row.preds if preds is None else preds[i]):
            go_id = terms[j]
            score *= 1 - alphas[go_rels.get_namespace(go_id)]
            if go_id in annots_dict:
                annots_dict[go_id] += score
            else:
                annots_dict[go_id] = score
        preds_w_diamond.append(annots_dict)
    return preds_w_diamond


def compute_prmetrics(
        labels, deep_preds, go_rels,
        ont="mf", verbose=False, out_dir="."):
    go_set = go_rels.get_namespace_terms(NAMESPACES[ont])
    go_set.remove(FUNC_DICT[ont])
    
    fmax = 0.0
    tmax = 0.0
    precisions = []
    recalls = []
    smin = 1000000.0
    rus = []
    mis = []
    for t in trange(0, 101):
        threshold = t / 100.0
        preds = []
        for i in range(len(deep_preds)):
            annots = set()
            for go_id, score in deep_preds[i].items():
                if score >= threshold:
                    annots.add(go_id)

            new_annots = set()
            for go_id in annots:
                new_annots |= go_rels.get_anchestors(go_id)
            preds.append(new_annots)
            
        # Filter classes
        preds = list(map(lambda x: set(filter(lambda y: y in go_set, x)), preds))
    
        fscore, prec, rec, s, ru, mi, fps, fns = evaluate_annotations(go_rels, labels, preds)
        avg_fp = sum(map(lambda x: len(x), fps)) / len(fps)
        avg_ic = sum(map(lambda x: sum(map(lambda go_id: go_rels.get_ic(go_id), x)), fps)) / len(fps)
        if(verbose):
            print(f'{avg_fp} {avg_ic}')
        precisions.append(prec)
        recalls.append(rec)
        #print(f'Fscore: {fscore}, Precision: {prec}, Recall: {rec} S: {s}, RU: {ru}, MI: {mi} threshold: {threshold}')
        if fmax < fscore:
            fmax = fscore
            tmax = threshold
        if smin > s:
            smin = s
    print(f'Fmax: {fmax:0.3f}, Smin: {smin:0.3f}, threshold: {tmax}')
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    sorted_index = np.argsort(recalls)
    recalls = recalls[sorted_index]
    precisions = precisions[sorted_index]
    aupr = np.trapz(precisions, recalls)
    print(f'AUPR: {aupr:0.3f}')
    plt.figure()
    lw = 2
    plt.plot(recalls, precisions, color='darkorange',
             lw=lw, label=f'AUPR curve (area = {aupr:0.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Area Under the Precision-Recall curve')
    plt.legend(loc="lower right")
    out_file = f"{out_dir}/aupr_{ont}.pdf"
    print(f"writing {out_file}")
    plt.savefig(out_file)

    df = pd.DataFrame({'precisions': precisions, 'recalls': recalls})
    out_file = f"{out_dir}/PR_{ont}.tab"
    print(f"writing {out_file}")
    df.to_csv(out_file, sep='\t')


def compute_roc(labels, preds):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)
    return roc_auc

def compute_mcc(labels, preds):
    # Compute ROC curve and ROC area for each class
    mcc = matthews_corrcoef(labels.flatten(), preds.flatten())
    return mcc

def evaluate_annotations(go, real_annots, pred_annots):
    total = 0
    p = 0.0
    r = 0.0
    p_total= 0
    ru = 0.0
    mi = 0.0
    fps = []
    fns = []
    for i in range(len(real_annots)):
        if len(real_annots[i]) == 0:
            continue
        tp = set(real_annots[i]).intersection(set(pred_annots[i]))
        fp = pred_annots[i] - tp
        fn = real_annots[i] - tp
        for go_id in fp:
            mi += go.get_ic(go_id)
        for go_id in fn:
            ru += go.get_ic(go_id)
        fps.append(fp)
        fns.append(fn)
        tpn = len(tp)
        fpn = len(fp)
        fnn = len(fn)
        total += 1
        recall = tpn / (1.0 * (tpn + fnn))
        r += recall
        if len(pred_annots[i]) > 0:
            p_total += 1
            precision = tpn / (1.0 * (tpn + fpn))
            p += precision
    ru /= total
    mi /= total
    r /= total
    if p_total > 0:
        p /= p_total
    f = 0.0
    if p + r > 0:
        f = 2 * p * r / (p + r)
    s = math.sqrt(ru * ru + mi * mi)
    return f, p, r, s, ru, mi, fps, fns

# Also load the DeepGOPlus evaluation code before continuing on
# use the UDSMProt evaluate_deepgoplus function:
# https://github.com/nstrodt/UDSMProt/blob/e68d784ae4d3c6f6a2ee39006a83d08402d9a495/code/utils/evaluate_deepgoplus.py
sys.path.append("/home/jlaw/projects/2020-01-deepgreen/models-to-compare/UDSMProt/code/utils")
from evaluate_deepgoplus import Ontology as UDSMProt_Ontology


def main(
        checkpoint_file, data_dir, fairseq_uniparc_dir,
        out_dir=None, combine_diamond_scores=False):

    obo_file = f"{fairseq_uniparc_dir}/inputs/deepgoplus/go.obo.gz"
    terms_file = f"{fairseq_uniparc_dir}/inputs/deepgoplus/terms.csv.gz"
    ont_obj = Ontology(obo_file=obo_file, restrict_terms_file=terms_file)
    _ancestor_array = ont_obj.ancestor_array()

    gofile = f"{fairseq_uniparc_dir}/inputs/deepgoplus/go.obo"
    go_rels = UDSMProt_Ontology(gofile, with_rels=True)


    # version = "2021_06_deepgoplus_esm1b_t33_n10"
    #version = "2021_06_deepgoplus_esm1_t6"

    #checkpoint_dir = f"/projects/deepgreen/jlaw/fairseq_go_checkpoints/{version}/"
    #checkpoint = "checkpoint_last.pt"
    #checkpoint_file = f"{checkpoint_dir}/{checkpoint}"
    #data_dir = "/projects/deepgreen/jlaw/fairseq_goa_splits/2021_06_fairseq_deepgoplus/"
    checkpoint_dir = os.path.dirname(checkpoint_file)
    checkpoint = os.path.basename(checkpoint_file)

    model = load_esm_model(checkpoint_dir, data_dir, checkpoint)

    inputs_filename = f'{data_dir}/input0/valid-2.raw.gz'
    print("\nMaking predictions")
    pred = make_predictions(model, inputs_filename, _ancestor_array)

    # Since the evaluation below is pretty slow,
    # clear the GPU memory for other processes
    del model
    torch.cuda.empty_cache()

    # Now run the deepgoplus evaluation

    deepgoplus_dir = "/projects/deepgreen/jlaw/inputs/deepgoplus/deepgoplus_data_2016"
    terms_file = f"{deepgoplus_dir}/terms.pkl"
    train_data_file = f"{deepgoplus_dir}/train_data.pkl"
    test_data_file = f"{deepgoplus_dir}/test_data.pkl"
    if out_dir is None:
        out_dir = f"{checkpoint_dir}/viz"
    os.makedirs(out_dir, exist_ok=True)

    terms_df = pd.read_pickle(terms_file)
    terms = terms_df['terms'].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}

    print(f"reading {train_data_file}")
    train_df = pd.read_pickle(train_data_file)
    print(f"reading {test_data_file}")
    test_df = pd.read_pickle(test_data_file)
    print(test_df.head(3))
    annotations = train_df['annotations'].values
    annotations = list(map(lambda x: set(x), annotations))
    test_annotations = test_df['annotations'].values
    test_annotations = list(map(lambda x: set(x), test_annotations))
    go_rels.calculate_ic(annotations + test_annotations)

    # pred_file = f"{deepgoplus_dir}/predictions.pkl"
    # test_df = pd.read_pickle(pred_file)

    # Print IC values of terms
    ics = {}
    for term in terms:
        ics[term] = go_rels.get_ic(term)

    if combine_diamond_scores:
        print("Combining predicted scores with diamond scores")
        pred = combine_with_diamond(pred, terms, go_rels, train_df, test_df, deepgoplus_dir, annotations)

    for ont in ['bp', 'mf', 'cc']:
        print("----"*20)
        print(f"Evaluating {ont}")
        # DeepGOPlus
        go_set = go_rels.get_namespace_terms(NAMESPACES[ont])
        go_set.remove(FUNC_DICT[ont])
        labels = test_df['annotations'].values
        labels = list(map(lambda x: set(filter(lambda y: y in go_set, x)), labels))

        if combine_diamond_scores:
            preds_for_deepgoplus_eval = pred
        else:
            # now make the annots_dict using our predictions
            preds_for_deepgoplus_eval = []
            for row in pred:
                # TODO there could be an error here
                # make sure to limit the terms to the current hierarchy
                annots_dict = {t: row[i] for i, t in enumerate(ont_obj.terms)}
                preds_for_deepgoplus_eval.append(annots_dict)

        print("Evaluating scores")
        compute_prmetrics(
                labels, preds_for_deepgoplus_eval, go_rels,
                ont=ont, out_dir=out_dir)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run the deepgoplus evaluation')
    parser.add_argument('--esm-checkpoint', type=str, help='ESM checkpoint file')
    parser.add_argument('--data-dir', type=str, help='Directory containing the fairseq data used to train/validate')
    parser.add_argument('--out-dir', type=str, help='Dir to store output log/viz files. Default is "viz" dir inside the checkpoint file\'s directory')
    parser.add_argument('--fairseq-uniparc-dir', type=str, 
            default="/home/jlaw/projects/2020-01-deepgreen/fairseq_uniparc_fork/",
            help='base dir of the fairseq-uniparc repo used to train the model')
    parser.add_argument('--combine-diamond-scores', '-D', action='store_true', default=False,
                        help="Combine the predicted scores with those from diamond as was done for UDSMProt")

    args = parser.parse_args()

    main(
        args.esm_checkpoint, args.data_dir, args.fairseq_uniparc_dir,
        args.out_dir, combine_diamond_scores=args.combine_diamond_scores)
