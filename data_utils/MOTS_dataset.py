import os
import numpy as np
import time

splits = ["train", "valid", "test"]
mapped_content = {0: 0.242607828674502385, 1: 0.542607828674502385, 2: 0.542607828674502385}

thresh_lis = [-500, 0, 325, 1500]
norm_lis = [0, 0.1, 0.9, 1]


#seed = 100

def task_index(name):
    if "liver" in name:
        return 0
    if "case" in name:
        return 1
    if "hepa" in name:
        return 2
    if "pancreas" in name:
        return 3
    if "colon" in name:
        return 4
    if "lung" in name:
        return 5
    if "spleen" in name:
        return 6

class MOTSDataset():
    def __init__(self, root, sample_points=8192, split='train', with_remission=False, iterate=10, remission=1,
                 should_map=False):
        self.root = root
        assert split in splits
        self.split = split
        self.sample_points = sample_points
        self.iterate = iterate
        self.with_remission = with_remission
        self.should_map = should_map
        self.remission = remission


        self.points_name = []
        self.label_name = []
        if split == 'train':
            self.root = os.path.join(self.root, 'train/pointcloud_0.5')
        else:
            self.root = os.path.join(self.root, 'test/pointcloud_0.5')
        points_path = os.path.join(self.root, 'scans')
        label_path = os.path.join(self.root, 'labels')
        seq_points_name = [os.path.join(points_path, pn) for pn in os.listdir(points_path) if pn.endswith('.bin')]
        seq_label_name = [os.path.join(label_path, ln) for ln in os.listdir(label_path) if ln.endswith('.label')]
        assert len(seq_points_name) == len(seq_label_name)
        self.indices = [[] for _ in range(len(seq_points_name))]
        self.points_name.extend(seq_points_name*iterate)
        self.label_name.extend(seq_label_name*iterate)
        self.points_name.sort()
        self.label_name.sort()
        if self.with_remission:
            self.feature_name = []
            feature_path = os.path.join(self.root, 'features_original')
            seq_feature_name = [os.path.join(feature_path, fn) for fn in os.listdir(feature_path) if fn.endswith('.bin')]
            self.feature_name.extend(seq_feature_name*iterate)
            self.feature_name.sort()


        label_weights_dict = mapped_content
        num_keys = len(label_weights_dict.keys())
        self.label_weights_lut = np.zeros((num_keys), dtype=np.float32)
        self.label_weights_lut[list(label_weights_dict.keys())] = list(label_weights_dict.values())
        self.label_weights_lut = np.power(np.amax(self.label_weights_lut[1:]) / self.label_weights_lut, 1 / 3.0)

        if should_map:
            remapdict = self.config["learning_map"]
            # make lookup table for mapping
            maxkey = max(remapdict.keys())
            # +100 hack making lut bigger just in case there are unknown labels
            self.remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
            self.remap_lut[list(remapdict.keys())] = list(remapdict.values())

    def __getitem__(self, index):
        points_name, label_name = self.points_name[index], self.label_name[index]
        task_id = task_index(points_name)
        base_name = os.path.basename(points_name)
        scan = np.fromfile(points_name, dtype=np.int64)
        points = scan.reshape((-1, 3))
        label = np.fromfile(label_name, dtype=np.uint8)
        label = label.reshape((-1))


        if self.should_map:
            label = self.remap_lut[label]
        label_weights = self.label_weights_lut[label]

        item = index // self.iterate
        indices = self.indices[item]
        remainder = np.arange(len(label))
        remainder = np.delete(remainder, indices)


        if len(remainder) > self.sample_points:
            choice = np.random.choice(remainder, self.sample_points, replace=False)
        else:
            choice = np.random.choice(len(label), self.sample_points-len(remainder), replace=True)
            choice = np.concatenate((remainder, choice))
            np.random.shuffle(choice)
        if len(indices) < len(label):
            indices.extend(choice)
            indices.sort()
            self.indices[item] = indices
        point_set = np.power(points[choice, :].astype(np.float32), 2)
        original_points = points[choice, :]
        semantic_seg = label[choice]
        mask = np.ones((len(label)), dtype=bool)


        coordmax = np.max(point_set[:, :], axis=0)
        coordmin = np.min(point_set[:, :], axis=0)
        coorditer = coordmax-coordmin
        center = point_set[np.random.choice(self.sample_points)]
        point_set[:, 0] = (point_set[:, 0] - center[0]) / coorditer[0]
        point_set[:, 1] = (point_set[:, 1] - center[1]) / coorditer[1]
        point_set[:, 2] = (point_set[:, 2] - center[2]) / coorditer[2]
        if self.with_remission:
            feature_name = self.feature_name[index]
            feature = np.fromfile(feature_name, dtype=np.float32)
            if('colon' in feature_name or 'lung' in feature_name):
                feature = feature.reshape((-1, 2))
                feature = np.insert(feature, 1, 0, axis=1)

            elif('spleen' in feature_name):
                feature = feature.reshape((-1, 2))
                feature = np.insert(feature, 2, 0, axis=1)
            else:
                feature = feature.reshape((-1, self.remission))
            feature_set = feature[choice, :]
            norm_feature = np.zeros((self.sample_points, 4))
            norm_feature[:, 0] = feature_set[:, 0]
            norm_feature[:, 1] = feature_set[:, 1]
            norm_feature[:, 2] = feature_set[:, 2]
            norm_feature[:, 3] = self.img_multi_thresh_normalized(feature_set[:, 0], thresh_lis, norm_lis)
            point_set = np.concatenate((point_set, norm_feature), axis=1)

        mask = mask[choice]
        sample_weight = label_weights[semantic_seg]
        sample_weight *= mask


        return point_set, semantic_seg, original_points, base_name, task_id

    def img_multi_thresh_normalized(self, feature, thresh_lis, norm_lis):
        new_feature = np.zeros_like(feature).astype(np.float)
        for i in range(1, len(thresh_lis)):
            mask = np.where((feature < thresh_lis[i]) & (feature >= thresh_lis[i-1]))
            k = (norm_lis[i] - norm_lis[i-1]) / (thresh_lis[i] - thresh_lis[i-1])
            b = norm_lis[i-1]
            new_feature[mask] = feature[mask] - thresh_lis[i-1]
            new_feature[mask] = k * new_feature[mask] + b
        new_feature[np.where(feature >= thresh_lis[-1])] = norm_lis[-1]
        return new_feature

    def truncate(self, CT):
        min_HU = -325
        max_HU = 325
        subtract = 0
        divide = 325.

        # truncate
        CT[np.where(CT <= min_HU)] = min_HU
        CT[np.where(CT >= max_HU)] = max_HU
        CT = CT - subtract
        CT = CT / divide
        return CT

    def __len__(self):
        return len(self.points_name)




def rotate(pointcloud):
    angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(angle)
    sinval = np.sin(angle)
    rotation_matrix = np.array([[cosval, -sinval, 0], [sinval, cosval, 0], [0, 0, 1]])  # R_z

    pointcloud = np.dot(pointcloud, rotation_matrix).astype(np.float32)  # todo handle possible additional features

    return pointcloud


def jitter(pointcloud):

    std = 0.01
    clip = 0.03

    jitter = np.random.normal(0.0, std, pointcloud.shape)
    jitter = np.clip(jitter, -clip, clip)

    pointcloud[:, 0:2] += jitter[:, 0:2]

    return pointcloud




