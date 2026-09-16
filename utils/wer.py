import os
import tensorflow as tf
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm


WER_COST_DEL = 3
WER_COST_INS = 3
WER_COST_SUB = 4

def wer_list(references, hypotheses):
    total_error = total_del = total_ins = total_sub = total_ref_len = 0

    for r, h in zip(references, hypotheses):
        res = wer_single(r=r, h=h)
        total_error += res["num_err"]
        total_del += res["num_del"]
        total_ins += res["num_ins"]
        total_sub += res["num_sub"]
        total_ref_len += res["num_ref"]

    wer = (total_error / total_ref_len) * 100
    del_rate = (total_del / total_ref_len) * 100
    ins_rate = (total_ins / total_ref_len) * 100
    sub_rate = (total_sub / total_ref_len) * 100

    return {
        "wer": wer,
        "del": del_rate,
        "ins": ins_rate,
        "sub": sub_rate,
    }


def wer_single(r, h):
    r = r.strip().split()
    h = h.strip().split()
    edit_distance_matrix = edit_distance(r=r, h=h)
    alignment, alignment_out = get_alignment(r=r, h=h, d=edit_distance_matrix)
    num_cor = np.sum([s == "C" for s in alignment])
    num_del = np.sum([s == "D" for s in alignment])
    num_ins = np.sum([s == "I" for s in alignment])
    num_sub = np.sum([s == "S" for s in alignment])
    num_err = num_del + num_ins + num_sub
    num_ref = len(r)

    return {
        "alignment": alignment,
        "alignment_out": alignment_out,
        "num_cor": num_cor,
        "num_del": num_del,
        "num_ins": num_ins,
        "num_sub": num_sub,
        "num_err": num_err,
        "num_ref": num_ref,
    }


def edit_distance(r, h):
    d = np.zeros((len(r) + 1) * (len(h) + 1), dtype=np.uint8).reshape(
        (len(r) + 1, len(h) + 1)
    )
    for i in range(len(r) + 1):
        for j in range(len(h) + 1):
            if i == 0:
                # d[0][j] = j
                d[0][j] = j * WER_COST_INS
            elif j == 0:
                d[i][0] = i * WER_COST_DEL
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitute = d[i - 1][j - 1] + WER_COST_SUB
                insert = d[i][j - 1] + WER_COST_INS
                delete = d[i - 1][j] + WER_COST_DEL
                d[i][j] = min(substitute, insert, delete)
    return d



def get_alignment(r, h, d):
    x = len(r)
    y = len(h)
    max_len = 3 * (x + y)

    alignlist = []
    align_ref = ""
    align_hyp = ""
    alignment = ""

    while True:
        if (x <= 0 and y <= 0) or (len(alignlist) > max_len):
            break
        elif x >= 1 and y >= 1 and d[x][y] == d[x - 1][y - 1] and r[x - 1] == h[y - 1]:
            align_hyp = " " + h[y - 1] + align_hyp
            align_ref = " " + r[x - 1] + align_ref
            alignment = " " * (len(r[x - 1]) + 1) + alignment
            alignlist.append("C")
            x = max(x - 1, 0)
            y = max(y - 1, 0)
        elif x >= 1 and y >= 1 and d[x][y] == d[x - 1][y - 1] + WER_COST_SUB:
            ml = max(len(h[y - 1]), len(r[x - 1]))
            align_hyp = " " + h[y - 1].ljust(ml) + align_hyp
            align_ref = " " + r[x - 1].ljust(ml) + align_ref
            alignment = " " + "S" + " " * (ml - 1) + alignment
            alignlist.append("S")
            x = max(x - 1, 0)
            y = max(y - 1, 0)
        elif y >= 1 and d[x][y] == d[x][y - 1] + WER_COST_INS:
            align_hyp = " " + h[y - 1] + align_hyp
            align_ref = " " + "*" * len(h[y - 1]) + align_ref
            alignment = " " + "I" + " " * (len(h[y - 1]) - 1) + alignment
            alignlist.append("I")
            x = max(x, 0)
            y = max(y - 1, 0)
        else:
            align_hyp = " " + "*" * len(r[x - 1]) + align_hyp
            align_ref = " " + r[x - 1] + align_ref
            alignment = " " + "D" + " " * (len(r[x - 1]) - 1) + alignment
            alignlist.append("D")
            x = max(x - 1, 0)
            y = max(y, 0)

    align_ref = align_ref[1:]
    align_hyp = align_hyp[1:]
    alignment = alignment[1:]

    return (
        alignlist[::-1],
        {"align_ref": align_ref, "align_hyp": align_hyp, "alignment": alignment},
    )



def cslr_beam_decode(logits, word_to_id, lengths, beam_width=10):
    logits = np.array(logits)
    T, B, V = logits.shape

    id_to_word = {v: k for k, v in word_to_id.items()}
    logits_tf = tf.convert_to_tensor(logits, dtype=tf.float32)
    log_probs = tf.nn.log_softmax(logits_tf, axis=-1)

    decoded_sparse, _ = tf.nn.ctc_beam_search_decoder(
        inputs=log_probs,
        sequence_length=tf.convert_to_tensor(lengths, dtype=tf.int32),
        beam_width=beam_width
    )

    dense = tf.sparse.to_dense(decoded_sparse[0], default_value=-1)

    results = []
    for b in range(B):
        seq = []
        for idx in dense[b].numpy():
            if idx == -1:
                continue
            word = id_to_word.get(int(idx), "")
            if word != "-":  # skip CTC blank
                seq.append(word)
        results.append(" ".join(seq))

    return results

if __name__ == "__main__":
    # ------------------ Load gloss dictionary ------------------
    gloss_dict_path = "/Users/gufran/Developer/Projects/AI/ERL-CSLR/assets/CSL-Daily/gloss_dict.npy"
    gloss_dict = np.load(gloss_dict_path, allow_pickle=True).item()

    # Keep only the first element if value is a list
    for k, v in gloss_dict.items():
        gloss_dict[k] = v[0]

    gloss_dict["-"] = 0  # blank token is at index 0

    # ------------------ Load logits ------------------
    logits_path = "/Users/gufran/Developer/Projects/AI/ERL-CSLR/data/CSL-Daily/sf_logits/test"
    logits_files = sorted(os.listdir(logits_path))  # ensure consistent order
    logits_files_ids = [l.replace(".npy", "") for l in logits_files]

    logits_list = []
    lengths = []

    for f in logits_files:
        arr = np.load(os.path.join(logits_path, f), allow_pickle=True)
        tensor = torch.tensor(arr).squeeze(0)  # remove extra dim if present
        logits_list.append(tensor)
        lengths.append(tensor.shape[0])

    # Pad sequences along time dimension
    logits = pad_sequence(logits_list, batch_first=False, padding_value=0)  # [T, B, V]


    # ------------------ Load test info ------------------
    test_info_path = "/Users/gufran/Developer/Projects/AI/ERL-CSLR/assets/CSL-Daily/test_info_ml.npy"
    test_info = np.load(test_info_path, allow_pickle=True).item()

    labels = [test_info[k]['label'] for k in test_info.keys() if test_info[k]["fileid"] in logits_files_ids]


    # ------------------ Decode in batches ------------------
    batch_size = 31
    B = logits.shape[1]

    full_decoded = []
    for i in tqdm(range(0, B, batch_size)):
        end = min(i + batch_size, B)
        logits_batch = logits[:, i:end, :].numpy()
        lengths_batch = lengths[i:end]
        decoded = cslr_beam_decode(logits_batch, gloss_dict, lengths_batch, blank_index=0, beam_width=10)
        
        full_decoded.extend(decoded)


    # ------------------ Compute WER ------------------
    print(len(labels), len(full_decoded))
    average_wer = wer_list(labels, full_decoded)
    print(f"Average WER: {average_wer:.4f}")