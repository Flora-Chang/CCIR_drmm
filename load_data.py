import numpy as np
import json
from util import FLAGS

#suppose input train_file is a json file like {query_id:123, pos: {histograms:[[],[],[]], passage_id:, },neg:{histograms:[[], [], ...[]], passage: ,}idf:[]}
#suppose input dev_file is a json file like{query_id:123, query: ,passages: [label:  , passsage_id: , histograms:[[],[]...[], idf:[]]]}

class LoadTrainData(object):
    def __init__(self, data_path, batch_size=64):
        self.data_path = data_path
        self.batch_size = batch_size
        self.data = open(self.data_path, 'r').readlines()
        self.batch_index = 0
        print("len data: ", len(self.data))

    def translation(self, histograms):
        result = []
        for histogram in histograms:
            row = []
            for item in histogram:
                row.append(float(item))
            result.append(row)
        return result

    def next_batch(self):
        data = np.array(self.data)
        data_size = len(data)
        num_batches_per_epoch = int(data_size / self.batch_size) + 1
        print("training_set: ", data_size, num_batches_per_epoch)
        np.random.shuffle(data)

        while self.batch_index < num_batches_per_epoch \
                and (self.batch_index + 1) * self.batch_size <= data_size:
            batch_histograms = []
            batch_idfs = []
            start_index = self.batch_index * self.batch_size
            self.batch_index += 1
            end_index = min(self.batch_index * self.batch_size, data_size)
            batch_data = data[start_index:end_index]

            for line in batch_data:
                histograms = []
                line = json.loads(line)
                query_id = line['query_id']
                pos = line['pos']
                pos_histogram = pos['histograms']
                neg = line['neg']
                neg_histogram = neg['histograms']
                idf = line['idf']
                pos_histogram = self.translation(pos_histogram)
                histograms.append(pos_histogram)
                neg_histogram = self.translation(neg_histogram)
                histograms.append(neg_histogram)
                idf = [float(i) for i in idf]
                batch_histograms.append(histograms)
                batch_idfs.append(idf)

            yield batch_histograms, batch_idfs


class LoadTestData(object):
    def __init__(self, data_path, batch_size):
        self.index = 0
        self.data = open(data_path, 'r').readlines()
        self.data_size = len(self.data)
        self.batch_size = batch_size
        self.cnt = 0

    def translation(self, histograms):
        result = []
        for histogram in histograms:
            row = []
            for item in histogram:
                row.append(int(item))
            result.append(row)
        return result

    def next_batch(self):
        if self.batch_size == -1:
            self.batch_size = 200
            self.data_size = self.batch_size*5
        while (self.index ) * self.batch_size < self.data_size:
            if (self.index + 1) * self.batch_size <= self.data_size:
                batch_data = self.data[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            else:
                batch_data = self.data[self.index * self.batch_size: self.data_size]
            self.index += 1
            batch_query_ids = []
            batch_ans_ids = []
            batch_ans_labels = []
            batch_idfs = []
            batch_histograms = []

            for line in batch_data:
                self.cnt += 1
                line = json.loads(line)
                query_id = line['query_id']
                passages = line['passages']
                idf = line['idf']
                idf = [int(i) for i in idf]
                for p in passages:
                    ans_id = p['passage_id']
                    ans_label = p['label']
                    histogram = p['histograms']
                    histogram = self.translation(histogram)
                    batch_histograms.append(histogram)
                    batch_idfs.append(idf)
                    batch_query_ids.append(query_id)
                    batch_ans_ids.append(ans_id)
                    batch_ans_labels.append(ans_label)

            yield batch_histograms, batch_idfs, batch_query_ids, batch_ans_ids, batch_ans_labels
        print("self.cnt:", self.cnt)