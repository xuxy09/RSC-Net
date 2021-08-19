import numpy as np

import torch
import torch.nn as nn


class FeatQueue(nn.Module):
    def __init__(self,  max_queue_size=30000):
        super(FeatQueue, self).__init__()
        self.max_queue_size = max_queue_size


    def append(self, queue, feat):
        if isinstance(feat, np.ndarray):
            queue = np.concatenate([queue, feat], axis=0)
            queue_size = queue.shape[0]
        else:
            queue = torch.cat([queue, feat], dim=0)
            queue_size = queue.size(0)
        # check if exceet queue_size
        if queue_size > self.max_queue_size:
            queue = self.pop(queue, queue_size - self.max_queue_size)
        return queue

    def pop(self, queue, num_item):
        queue = queue[num_item:]
        return queue

    def update_queue_size(self, size):
        try:
            curr_queue_size = getattr(self, 'curr_queue_size')
        except:
            curr_queue_size = 0

        curr_queue_size += size
        if curr_queue_size >= self.max_queue_size:
            curr_queue_size = self.max_queue_size
        setattr(self, 'curr_queue_size', curr_queue_size)


    def update(self, name, feat):
        try:
            queue = getattr(self, name)
            queue = self.append(queue, feat)
            setattr(self, name, queue)
        except:
            setattr(self, name, feat)

    def update_all(self, feats, names):
        for name, feat in zip(names, feats):
            self.update(name, feat)

    def sample(self, name, indices):
        queue = getattr(self, name)
        # print("name:{}, indices:{}, queue shape {}".format(name, indices, queue.shape))
        out = queue[indices]
        return out

    def batch_sample(self, indices_list, name):
        list_items = []
        for indices in indices_list:
            list_items.append(self.sample(name, indices).unsqueeze(0))
        # print(name)
        list_items = torch.cat(list_items, dim=0)
        return list_items


    def batch_sample_all(self, indices_list, names):
        results = []
        for name in names:
            results.append(self.batch_sample(indices_list, name))
        return results

    def select_indices(self, dataset_names, dataset_indices, sample_size=8192):
        indices_list = []
        length_list = []
        for name, dataset_index in zip(dataset_names, dataset_indices):
            dataset_names = getattr(self, 'dataset_names')
            dataset_indices = getattr(self, 'dataset_indices')
            condition = np.logical_and(dataset_names == name, dataset_indices == dataset_index)
            same_indices = np.where(condition)[0]
            all_indices = np.arange(getattr(self, 'curr_queue_size'))
            if same_indices.size != 0:
                diff_indices = np.delete(all_indices, same_indices)
            else:
                diff_indices = all_indices

            # sample
            perm_index = np.random.choice(diff_indices, size=min(sample_size, len(diff_indices)), replace=False) # torch.randint(high=self.curr_queue_size, size=(min(sample_size, self.curr_queue_size),), dtype=torch.int32).tolist()

            indices_list.append(perm_index)
            length_list.append(len(perm_index))

        minimum_length = np.min(length_list)
        indices_list = [indices[:minimum_length] for indices in indices_list]
        return indices_list


if __name__ == '__main__':


    feat_queue = FeatQueue(max_queue_size=100)


