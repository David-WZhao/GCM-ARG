import torch.utils.data as tud
import pandas as pd
import torch

class ARGDataSet(tud.Dataset):
    def __init__(self, data):
        super(ARGDataSet, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return torch.FloatTensor(self.data.iloc[item]['seq_map']), self.data.iloc[item]['anti_label'], self.data.iloc[item]['mech_label'], self.data.iloc[item]['type_label']

class ARGDataLoader(object):
    def __init__(self):
        print("loading data...")
        self.anti_count, self.mech_count, self.type_count = 15, 6, 2

    def load_test_dataSet(self, batch_size):
        print('loading test data...')
        test_data = pd.read_pickle('./data/test/test.pickle')
        test_data = tud.DataLoader(ARGDataSet(test_data), batch_size=batch_size, shuffle=True, num_workers=0)
        return test_data

    def load_n_cross_data(self, k, batch_size):
        print('loading cross_' + str(k) + ' train_val data ...')
        train_data = pd.read_pickle('data/train_val/cross_' + str(k) + '_train.pickle')
        val_data = pd.read_pickle('data/train_val/cross_' + str(k) + '_val.pickle')
        train_data = tud.DataLoader(ARGDataSet(train_data), batch_size=batch_size, shuffle=True, num_workers=0)
        val_data = tud.DataLoader(ARGDataSet(val_data), batch_size=batch_size, shuffle=True, num_workers=0)
        return train_data, val_data

    def get_data_shape(self):
        return self.anti_count, self.mech_count, self.type_count

# if __name__ == '__main__':
    # dataloader = ARGDataLoader(batch_size=16, train_rate=0.8, K=5)
    # train_val_dataloader = dataloader.get_train_val_dataloader()
    # train_dataloader = train_val_dataloader[0]['train']
    # val_dataloader = train_val_dataloader[0]['val']
    # test_dataloader = dataloader.get_test_dataloader()
    # print(len(train_dataloader))
    # index, (seq_map, anti_label, mech_label, type_label) = next(enumerate(train_dataloader))
    # print(seq_map.size())
    # print(anti_label.size())
    # print(mech_label.size())
    # print(type_label.size())
