import tensorflow as tf
import pandas as pd
import numpy as np


def dcg_k(data, k=3):
    score = 0
    for num in range(k):
        top = np.power(2, data['label'][num]) - 1
        bottom = np.log2(data['rank'][num] + 1)
        score += np.divide(top, bottom)
    return score


def normalized_dcg_k(data, real_data, k=3):
    score = 0
    real_score = 0
    for num in range(k):
        top = np.power(2, data['label'][num]) - 1
        bottom = np.log2(data['rank'][num] + 1)
        score += np.divide(top, bottom)
    for num in range(k):
        top = np.power(2, real_data['label'][num]) - 1
        bottom = np.log2(real_data['rank'][num] + 1)
        real_score += np.divide(top, bottom)
    if real_score:
        return np.divide(score, real_score)
    else:
        return score

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=-1)

def test(sess, model, testing_set, filename=None):
    Norm_DCG3 = []
    Norm_DCG5 = []
    Norm_FULL =[]
    DCG_3 = []
    DCG_5 = []
    DCG_full = []

    little_list = []
    little_dcg = []

    zero_3 = 0
    zero_5 = 0
    for batch_data in testing_set.next_batch():
        histograms, idfs, query_ids, answers_ids, answers_label = batch_data

        fd = {model.histogram: histograms,
              model.idf: idfs}

        res = sess.run([model.score], fd)

        res = list(zip(query_ids, answers_ids, answers_label, res[0].tolist()))

        unique_query_ids = list(set(query_ids))
        df = pd.DataFrame(res, columns=['query_id', 'passage_id', 'label', 'score'])

        out_frames = []
        for query_id in unique_query_ids:
            passages = df[df['query_id'] == query_id]

            #is it correct?
            #passages['score'] = softmax(passages['score'])
            rank = range(1, passages.count()['label'] + 1)

            # result = passages.sort(['score'], ascending=False).reset_index(drop=True)
            real = passages.sort_values(by=['label'], ascending=False).reset_index(drop=True)
            real['rank'] = rank
            real.drop('score', axis=1, inplace=True)
            result = passages.sort_values(by=['score'], ascending=False).reset_index(drop=True)
            result['rank'] = rank
            #result.drop('score', axis=1, inplace=True)

            dcg_3 = dcg_k(result, 3)
            dcg_5 = dcg_k(result, 5)
            dcg_full = dcg_k(result, rank[-1])
            norm_dcg_3 = normalized_dcg_k(result, real, 3)
            norm_dcg_5 = normalized_dcg_k(result, real, 5)
            norm_dcg_full = normalized_dcg_k(result, real, rank[-1])

            if norm_dcg_3 < 0.65 and norm_dcg_5 < 0.70:
                little_list.append(query_id)
                little_dcg.append(norm_dcg_3)

            result = passages.sort_values(by=['passage_id'], ascending=True).reset_index(drop=True)
            out_frames.append(result)

            if dcg_5 < 0.1:
                zero_3 += 1
                zero_5 += 1
            elif dcg_3 < 0.1:
                zero_3 += 1

            Norm_DCG3.append(norm_dcg_3)
            Norm_DCG5.append(norm_dcg_5)
            Norm_FULL.append(norm_dcg_full)

            DCG_3.append(dcg_3)
            DCG_5.append(dcg_5)
            DCG_full.append(dcg_full)
        if filename is not None:
            out_df = pd.concat(out_frames)
            out_df.to_csv("../result/" + filename, columns = ['query_id', 'passage_id', 'label', 'score'], index=False)

    dcg_3_mean = np.mean(DCG_3)
    dcg_5_mean = np.mean(DCG_5)
    dcg_full_mean = np.mean(DCG_full)

    norm_dcg_3_mean = np.mean(Norm_DCG3)
    norm_dcg_5_mean = np.mean(Norm_DCG5)
    norm_dcg_full_mean = np.mean(Norm_FULL)

    print("number of Zero DCG@3: ", zero_3)
    print("number of Zero DCG@5: ", zero_5)
    print("DCG@3 Mean: ", dcg_3_mean, "\tNorm: ", norm_dcg_3_mean)
    print("DCG@5 Mean: ", dcg_5_mean, "\tNorm: ", norm_dcg_5_mean)
    print("DCG@full Mean: ", dcg_full_mean, "\tNorm: ", norm_dcg_full_mean)
    print("=" * 60)

    with open("./worse_queries.txt", 'w') as f:
        for i, j in zip(little_list, little_dcg):
            f.write(str(i) + '\t' + str(j) + '\n')

    return dcg_3_mean, dcg_5_mean, dcg_full_mean

