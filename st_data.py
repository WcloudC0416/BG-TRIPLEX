
import os
from glob import glob
import warnings

warnings.filterwarnings('ignore')

import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from PIL import ImageFile, Image
Image.MAX_IMAGE_PIXELS = None  # 完全解除限制
import torch
import torchvision
import torchvision.transforms as transforms
import scprep as scp
import pyvips as pv
import cv2

from utils import smooth_exp

ImageFile.LOAD_TRUNCATED_IMAGES = True #配置选项，用于控制库在加载被截断的图像文件时的行为。如果设置为 True，那么即使图像文件不完整或损坏，库也会尝试加载它。这可能会导致图像加载不完全，但至少可以访问文件中的部分数据。
Image.MAX_IMAGE_PIXELS = None


class BaselineDataset(torch.utils.data.Dataset):
    """Some Information about baselines"""

    def __init__(self):
        super(BaselineDataset, self).__init__()

        self.train_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            torchvision.transforms.RandomApply([torchvision.transforms.RandomRotation((90, 90))]),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        self.test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        self.features_train_transforms = transforms.Compose([
            transforms.ToPILImage(),
            torchvision.transforms.RandomApply([transforms.RandomRotation((0, 180))]),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])

        self.features_test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])


    def get_img(self, name: str):
        """Load whole slide image of a sample.

        Args:
            name (str): name of a sample

        Returns:
            PIL.Image: return whole slide image.
        """

        img_dir = self.data_dir + '/ST-imgs'
        if self.data == 'her2st':
            img_dir = self.data_dir + '/ST-imgs'
            pre = os.path.join(img_dir, name[0], name)
            fig_name = os.listdir(pre)
            for i in range(3):
                if fig_name[i].endswith("cat_image.jpg" or "seg.jpg"):
                    continue
                else:
                    figname = fig_name[i]
                    break
            path = pre + '/' + figname
            print(path)
        elif self.data == 'stnet' or '10x_breast' in self.data:
            print(name)
            path = glob(img_dir + '/*' + name + '.tif')[0]
        elif 'DRP' in self.data:
            path = glob(img_dir + '/*' + name + '.svs')[0]
        else:
            print(name)
            path = glob(img_dir + '/*' + name + '*' + '.jpg')[0]

        if self.use_pyvips:
            im = pv.Image.new_from_file(path, level=0)
        else:
            im = Image.open(path)
        return im

    def get_cnt(self, name: str):
        """Load gene expression data of a sample.

        Args:
            name (str): name of a sample

        Returns:
            pandas.DataFrame: return gene expression. 
        """
        path = self.data_dir + '/ST-cnts/' + name + '_sub.parquet'
        df = pd.read_parquet(path)

        return df

    def get_pos(self, name: str):
        """Load position information of a sample.
        The 'id' column is for matching against the gene expression table.

        Args:
            name (str): name of a sample

        Returns:
            pandas.DataFrame: return DataFrame with position information.
        """
        path = self.data_dir + '/ST-spotfiles/' + name + '_selection.tsv'
        df = pd.read_csv(path, sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i]) + 'x' + str(y[i]))
        df['id'] = id

        return df

    def get_meta(self, name: str):
        """Load both gene expression and postion data and merge them.

        Args:
            name (str): name of a sample

        Returns:
            pandas.DataFrame: return merged table (gene exp + position)
        """

        pos = self.get_pos(name)

        if 'DRP' not in self.data:
            cnt = self.get_cnt(name)
            meta = cnt.join(pos.set_index('id'), how='inner')
        else:
            meta = pos

        if self.mode == "external_test":
            meta = meta.sort_values(['x', 'y'])
        else:
            meta = meta.sort_values(['y', 'x'])

        return meta


class STDataset(BaselineDataset):
    """Dataset to load ST data for TRIPLEX
    """

    def __init__(self,
                 mode: str,
                 fold: int = 0,
                 extract_mode: str = None,
                 test_data=None,
                 **kwargs):
        """
        Args:
            mode (str): 'train', 'test', 'external_test', 'extraction', 'inference'.
            fold (int): Number of fold for cross validation.
            test_data (str, optional): Test data name. Defaults to None.
        """
        super().__init__()

        # Set primary attribute
        self.gt_dir = kwargs['t_global_dir']
        self.num_neighbors = kwargs['num_neighbors']
        self.neighbor_dir = f"{kwargs['neighbor_dir']}_{self.num_neighbors}_224"

        self.use_pyvips = kwargs['use_pyvips']

        self.r = kwargs['radius'] // 2
        self.extract_mode = False

        self.mode = mode

        if mode in ["external_test", "inference"]:
            self.data = test_data
            self.data_dir = f"{kwargs['data_dir']}/test/{self.data}"
        elif mode == "extraction":
            self.extract_mode = extract_mode
            self.data = test_data
            node_id = kwargs['node_id']
            self.data_dir = f"{kwargs['data_dir']}/test/{self.data}"
        else:
            self.data = kwargs['type']
            self.data_dir = f"{kwargs['data_dir']}/{self.data}"

        names = os.listdir(self.data_dir + '/ST-spotfiles')
        names.sort()
        names = [i.split('_selection.tsv')[0] for i in names]

        if mode in ["external_test", "inference", "train", "test"]:
            self.names = names
            ### @@Verified
            self.per_patch_csv_path = os.path.join(self.data_dir, 'per_patch.csv')
            self.n_patches_csv_path = os.path.join(self.data_dir, 'n_patches.csv')
            self.nuc_per_patch_csv_path = os.path.join(self.data_dir, 'nuc_per_patch.csv')
            self.nuc_n_patches_csv_path = os.path.join(self.data_dir, 'nuc_n_patches.csv')
            self.edge_per_patch_csv_path = os.path.join(self.data_dir, 'edge_per_patch.csv')
            self.edge_n_patches_csv_path = os.path.join(self.data_dir, 'edge_n_patches.csv')
            self.per_patch_csv = pd.read_csv(self.per_patch_csv_path)
            self.n_patches_csv = pd.read_csv(self.n_patches_csv_path)
            self.nuc_per_patch_csv = pd.read_csv(self.nuc_per_patch_csv_path)
            self.nuc_n_patches_csv = pd.read_csv(self.nuc_n_patches_csv_path)
            self.edge_per_patch_csv = pd.read_csv(self.edge_per_patch_csv_path)
            self.edge_n_patches_csv = pd.read_csv(self.edge_n_patches_csv_path)
            if self.mode == "external_test":
                self.per_patch_csv = self.per_patch_csv.sort_values(['x', 'y'])
                self.n_patches_csv = self.n_patches_csv.sort_values(['x', 'y'])
                self.nuc_per_patch_csv = self.nuc_per_patch_csv.sort_values(['x', 'y'])
                self.nuc_n_patches_csv = self.nuc_per_patch_csv.sort_values(['x', 'y'])
                self.edge_per_patch_csv = self.edge_per_patch_csv.sort_values(['x', 'y'])
                self.edge_n_patches_csv = self.edge_per_patch_csv.sort_values(['x', 'y'])
            else:
                self.per_patch_csv = self.per_patch_csv.sort_values(['y', 'x'])
                self.n_patches_csv = self.n_patches_csv.sort_values(['y', 'x'])
                self.nuc_per_patch_csv = self.nuc_per_patch_csv.sort_values(['y', 'x'])
                self.nuc_n_patches_csv = self.nuc_per_patch_csv.sort_values(['y', 'x'])
                self.edge_per_patch_csv = self.edge_per_patch_csv.sort_values(['y', 'x'])
                self.edge_n_patches_csv = self.edge_per_patch_csv.sort_values(['y', 'x'])
            ###
        elif mode == "extraction":
            # self.names = np.array_split(names, 2)[node_id]
            self.names = names
            if extract_mode == "neighbor":
                self.names = [name for name in self.names if
                              not os.path.exists(os.path.join(self.neighbor_dir, f"{name}.pt"))]
            elif extract_mode == "target":
                self.names = [name for name in self.names if
                              not os.path.exists(os.path.join(self.gt_dir, f"{name}.pt"))]

        else:
            if self.data == 'stnet':
                kf = KFold(8, shuffle=True, random_state=2021)
                patients = ['BC23209', 'BC23270', 'BC23803', 'BC24105', 'BC24220', 'BC23268', 'BC23269', 'BC23272',
                            'BC23277', 'BC23287', 'BC23288', 'BC23377', 'BC23450', 'BC23506', 'BC23508', 'BC23567',
                            'BC23810', 'BC23895', 'BC23901', 'BC23903', 'BC23944', 'BC24044', 'BC24223']
                patients = np.array(patients)
                _, ind_val = [i for i in kf.split(patients)][fold]
                paients_val = patients[ind_val]

                te_names = []
                for pp in paients_val:
                    te_names += [i for i in names if pp in i]

            elif self.data == 'her2st':
                patients = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
                te_names = [i for i in names if patients[fold] in i]
            elif self.data == 'skin':
                patients = ['P2', 'P5', 'P9', 'P10']
                te_names = [i for i in names if patients[fold] in i]

            tr_names = list(set(names) - set(te_names))

            if self.mode == 'train':
                self.names = tr_names
            else:
                self.names = te_names

        if self.use_pyvips:
            with open(f"{self.data_dir}/slide_shape.pickle", "rb") as f:
                self.img_shape_dict = pickle.load(f)
        else:
            pass
        

        self.meta_dict = {i: self.get_meta(i) for i in self.names}

        if mode not in ["extraction", "inference"]:
            gene_list = list(np.load(self.data_dir + f'/genes_{self.data}.npy', allow_pickle=True))
            self.exp_dict = {i: scp.transform.log(scp.normalize.library_size_normalize(m[gene_list])) for i, m in
                             self.meta_dict.items()}

            # Smoothing data 
            self.exp_dict = {i: smooth_exp(m).values for i, m in self.exp_dict.items()}
            print(self.exp_dict['10x_breast_ff1'].shape)

        if mode == "external_test":
            self.center_dict = {i: np.floor(m[['pixel_y', 'pixel_x']].values).astype(int) for i, m in
                                self.meta_dict.items()}
            self.loc_dict = {i: m[['x', 'y']].values for i, m in self.meta_dict.items()}
        else:
            self.center_dict = {i: np.floor(m[['pixel_x', 'pixel_y']].values).astype(int) for i, m in
                                self.meta_dict.items()}
            self.loc_dict = {i: m[['y', 'x']].values for i, m in self.meta_dict.items()}

        self.lengths = [len(i) for i in self.meta_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(self.names))

    def __getitem__(self, index):
        """Return one piece of data for training, and all data within a patient for testing.

        Returns:
            tuple: 
                patches (torch.Tensor): Target spot images
                edge_patches (torch.Tensor): Target spot edge images
                seg_patches (torch.Tensor): Target spot seg images
                neighbor_edge_patches (torch.Tensor): Neighbor edge images
                neighbor_seg_patches (torch.Tensor): Neighbor seg images
                exps (torch.Tensor): Gene expression of the target spot.
                pid (torch.LongTensor): patient index
                sid (torch.LongTensor): spot index
                wsi (torch.Tensor): Features extracted from all spots for the patient
                position (torch.LongTensor): Relative position of spots 
                neighbors (torch.Tensor): Features extracted from neighbor regions of the target spot.
                maks_tb (torch.Tensor): Masking table for neighbor features
        """
        if self.mode == 'train' or self.mode == 'test':

            i = 0
            while index >= self.cumlen[i]:
                i += 1
            idx = index
            if i > 0:
                idx = index - self.cumlen[i - 1]
            name = self.id2name[i]


            path = self.per_patch_csv.iloc[idx]['path']  # 单patch
            edge_path = self.per_patch_csv.iloc[idx]['path']
            nuc_path = self.nuc_per_patch_csv.iloc[idx]['path']
            n_path = self.n_patches_csv.iloc[idx]['path']
            n_edge_path = self.edge_n_patches_csv.iloc[idx]['path']
            n_nuc_path = self.nuc_n_patches_csv.iloc[idx]['path']
            im_t = Image.open(path)
            edge_t = Image.open(edge_path)
            nuc_t = Image.open(nuc_path)
            n_img_t = Image.open(n_path)
            n_edge_t = Image.open(n_edge_path)
            n_nuc_t = Image.open(n_nuc_path)
            im = np.array(im_t)
            edge_im = np.array(edge_t)
            nuc_im = np.array(nuc_t)
            n_im = np.array(n_img_t)
            n_edge_im = np.array(n_edge_t)
            n_nuc_im = np.array(n_nuc_t)
            im_t.close()
            edge_t.close()
            nuc_t.close()
            n_img_t.close()
            n_edge_t.close()
            n_nuc_t.close()


            center = self.center_dict[name][idx]

            print("training name", name)


            patches = []
            edge_patches = []
            nuc_patches = []
            n_patches = []
            n_edge_patches = []
            n_nuc_patches = []
            if self.mode == 'train':
                patch = self.train_transforms(im)
                edge_patch = self.train_transforms(edge_im)
                nuc_patch = self.train_transforms(nuc_im)
                n_patch = self.train_transforms(n_im)
                n_edge_patch = self.train_transforms(n_edge_im)
                n_nuc_patch = self.train_transforms(n_nuc_im)
            else:
                patch = self.test_transforms(im)
                edge_patch = self.test_transforms(edge_im)
                nuc_patch = self.test_transforms(nuc_im)
                n_patch = self.test_transforms(n_im)
                n_edge_patch = self.test_transforms(n_edge_im)
                n_edge_patches

            patches.append(patch)
            edge_patches.append(edge_patch)
            nuc_patches.append(nuc_patch)
            n_patches.append(n_patch)
            n_edge_patches.append(n_edge_patch)
            n_nuc_patches.append(n_nuc_patch)
            patches = torch.stack(patches, dim=0)
            edge_patches = torch.stack(edge_patches, dim=0)
            nuc_patches = torch.stack(nuc_patches, dim=0)
            n_patches = torch.stack(n_patches, dim=0)
            n_edge_patches = torch.stack(n_edge_patches, dim=0)
            n_nuc_patches = torch.stack(n_nuc_patches, dim=0)

            if self.mode == "train":
                exps = self.exp_dict[name][idx]
                exps = torch.Tensor(exps)
            else:
                exps = self.exp_dict[name]
                exps = torch.Tensor(exps)
            print("1111111111")
            sid = torch.LongTensor([idx])
            neighbors = torch.load(self.data_dir + f"/{self.neighbor_dir}/{name}.pt")[idx]
        else:
            i = index
            name = self.names[0]
            print('testset name: ', name)
            path = self.per_patch_csv.iloc[i]['path']   # 单patch
            neighbor_edge_file = self.n_patches_csv.iloc[i]['path']  # 邻域patches
            seg_file = self.nuc_per_patch_csv.iloc[i]['path']   # mask
            neighbor_seg_file = self.nuc_n_patches_csv.iloc[i]['path']  # 邻域mask
            im_t = Image.open(path)
            edge_img_t = Image.open(path).convert('L')              # TODO： 暂时用灰度图像代替
            neighbor_edge_img_t = Image.open(neighbor_edge_file).convert('L')
            seg_img_t = Image.open(seg_file).convert('L')
            neighbor_seg_img_t = Image.open(neighbor_seg_file).convert('L')

            im = np.array(im_t)
            edge_img = np.array(edge_img_t)
            seg_img = np.array(seg_img_t)
            neighbor_seg_img = np.array(neighbor_seg_img_t)
            neighbor_edge_img = np.array(neighbor_edge_img_t)

            im_t.close()
            edge_img_t.close()
            seg_img_t.close()
            neighbor_seg_img_t.close()
            neighbor_edge_img_t.close()

            if self.use_pyvips:
                img_shape = self.img_shape_dict[name]
            else:
                img_shape = im.shape
                # img_shape = seg_img.shape

            centers = self.center_dict[name]

            n_patches = len(centers)
            print("n_patches", n_patches)
            if self.extract_mode == 'neighbor':
                # patches = torch.zeros((n_patches, 3, 2 * self.r * self.num_neighbors, 2 * self.r * self.num_neighbors))
                # edge_patches = torch.zeros(
                #     (n_patches, 1, 2 * self.r * self.num_neighbors, 2 * self.r * self.num_neighbors))
                # seg_patches = torch.zeros(
                #     (n_patches, 1, 2 * self.r * self.num_neighbors, 2 * self.r * self.num_neighbors))
                patches = []
                edge_patches = []
                nuc_patches = []
            else:
                # patches = torch.zeros((n_patches, 3, 2 * self.r, 2 * self.r))
                # edge_patches = torch.zeros((n_patches, 1, 2 * self.r, 2 * self.r))
                # seg_patches = torch.zeros((n_patches, 1, 2 * self.r, 2 * self.r))
                # neighbor_edge_patches = torch.zeros((n_patches, 2 * self.r * self.num_neighbors, 2 * self.r * self.num_neighbors))
                # neighbor_seg_patches = torch.zeros((n_patches, 2 * self.r * self.num_neighbors, 2 * self.r * self.num_neighbors))
                patches = []
                edge_patches = []
                nuc_patches = []
                n_edge_patches = []
                n_nuc_patches = []


            patch = self.test_transforms(im)
            edge = self.features_test_transforms(edge_img)
            seg = self.features_test_transforms(seg_img)
            neighbor_edge = self.features_test_transforms(neighbor_edge_img)
            neighbor_nuc = self.features_test_transforms(neighbor_seg_img)
            patches.append(patch)
            edge_patches.append(edge)
            nuc_patches.append(seg)
            n_edge_patches.append(neighbor_edge)
            n_nuc_patches.append(neighbor_nuc)

            patches = torch.stack(patches, dim=0)
            edge_patches = torch.stack(edge_patches, dim=0)
            seg_patches = torch.stack(nuc_patches, dim=0)
            n_edge_patches = torch.stack(n_edge_patches, dim=0)
            n_nuc_patches = torch.stack(n_seg_patches, dim=0)
            mask_tb = torch.ones((1, self.num_neighbors ** 2))

            if self.mode == "extraction":
                return patches, edge_patches, seg_patches

            if self.mode != "inference":
                exps = self.exp_dict[name]
                exps = torch.Tensor(exps)

            sid = torch.arange(n_patches)
            print(sid)
            neighbors = torch.load(self.data_dir + f"/{self.neighbor_dir}/{name}.pt")

        wsi = torch.load(self.data_dir + f"/{self.gt_dir}/{name}.pt")

        pid = torch.LongTensor([i])
        pos = self.loc_dict[name]
        position = torch.LongTensor(pos)

        if self.mode not in ["external_test", "inference"]:
            name += f"+{self.data}"

        if self.mode == 'train' or 'test':
            return patches, edge_patches, nuc_patches, n_edge_patches, n_nuc_patches, exps, pid, sid, wsi, position, neighbors, mask_tb

        elif self.mode == 'inference':
            return patches, edge_patches, nuc_patches, n_edge_patches, n_nuc_patches, sid, wsi, position, neighbors, mask_tb

        else:
            print(sid)
            return patches, edge_patches, nuc_patches, n_edge_patches, n_nuc_patches, exps, sid, wsi, position, name, neighbors, mask_tb

    def __len__(self):
        if self.mode == 'train':
            return self.cumlen[-1]
        else:
            if '10x_breast' in self.names[0]:
                return len(self.meta_dict[self.names[0]])
            return len(self.meta_dict)

    def make_masking_table(self, x: int, y: int, img_shape: tuple):
        """Generate masking table for neighbor encoder.

        Args:
            x (int): x coordinate of target spot
            y (int): y coordinate of target spot
            img_shape (tuple): Shape of whole slide image

        Raises:
            Exception: if self.num_neighbors is bigger than 5, raise error.

        Returns:
            torch.Tensor: masking table
        """

        # Make masking table for neighbor encoding module
        mask_tb = torch.ones(self.num_neighbors ** 2)

        def create_mask(ind, mask_tb, window):
            if y - self.r * window < 0:
                mask_tb[self.num_neighbors * ind:self.num_neighbors * ind + self.num_neighbors] = 0
            if y + self.r * window > img_shape[0]:
                mask_tb[(self.num_neighbors ** 2 - self.num_neighbors * (ind + 1)):(
                            self.num_neighbors ** 2 - self.num_neighbors * ind)] = 0
            if x - self.r * window < 0:
                mask = [i + ind for i in range(self.num_neighbors ** 2) if i % self.num_neighbors == 0]
                mask_tb[mask] = 0
            if x + self.r * window > img_shape[1]:
                mask = [i - ind for i in range(self.num_neighbors ** 2) if
                        i % self.num_neighbors == (self.num_neighbors - 1)]
                mask_tb[mask] = 0

            return mask_tb

        ind = 0
        window = self.num_neighbors
        while window >= 3:
            mask_tb = create_mask(ind, mask_tb, window)
            ind += 1
            window -= 2

        return mask_tb

    def extract_patches_pyvips(self, slide, x: int, y: int, img_shape: tuple):
        tile_size = self.r * 2
        expansion_size = tile_size * self.num_neighbors
        padding_color = 255

        x_lt = x - tile_size * 2
        y_lt = y - tile_size * 2
        x_rd = x + tile_size * 3
        y_rd = y + tile_size * 3

        # Determine if padding is needed and calculate padding amounts
        x_left_pad = max(0, -x_lt)
        x_right_pad = max(0, x_rd - img_shape[1])
        y_up_pad = max(0, -y_lt)
        y_down_pad = max(0, y_rd - img_shape[0])

        # Adjust coordinates and dimensions
        x_lt = max(x_lt, 0)
        y_lt = max(y_lt, 0)
        width = min(x_rd, img_shape[1]) - x_lt
        height = min(y_rd, img_shape[0]) - y_lt

        # Extract and convert image
        im = slide.extract_area(x_lt, y_lt, width, height)
        im = np.array(im)[:, :, :3]

        # Check if any padding is necessary
        if x_left_pad or x_right_pad or y_up_pad or y_down_pad:
            # Create a full image with padding where necessary
            padded_image = np.full((expansion_size, expansion_size, 3), padding_color, dtype='uint8')

            # Calculate the placement indices for the image within the padded array
            start_x = x_left_pad
            end_x = x_left_pad + width
            start_y = y_up_pad
            end_y = y_up_pad + height

            # Place the image within the padded area
            padded_image[start_y:end_y, start_x:end_x] = im
            image = padded_image
        else:
            image = im

        return image
        
    def ensure_size(self, img, size):
        """Ensure the image has the correct size."""
        if img.shape[1] != size[1] or img.shape[0] != size[0]:
            img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
        return img
            
    def extract_features_patches_pyvips(self, slide, x: int, y: int, img_shape: tuple):
        tile_size = self.r * 2
        expansion_size = tile_size * self.num_neighbors
        padding_color = 0  # 灰度图中的白色

        x_lt = x - tile_size
        y_lt = y - tile_size
        x_rd = x + tile_size
        y_rd = y + tile_size

        # 计算需要的填充量
        x_left_pad = max(0, -x_lt)
        x_right_pad = max(0, x_rd - img_shape[1])
        y_up_pad = max(0, -y_lt)
        y_down_pad = max(0, y_rd - img_shape[0])

        # 调整坐标和尺寸
        x_lt = max(x_lt, 0)
        y_lt = max(y_lt, 0)
        width = min(x_rd, img_shape[1]) - x_lt
        height = min(y_rd, img_shape[0]) - y_lt

        # 提取图像区域
        im = slide.extract_area(x_lt, y_lt, width, height, band=True)  # band=True 表示提取单通道图像
        im = np.array(im)  # 直接转换为灰度图

        # 检查是否需要填充
        if x_left_pad or x_right_pad or y_up_pad or y_down_pad:
            # 创建一个全尺寸的填充图像
            padded_image = np.full((expansion_size, expansion_size), padding_color, dtype='uint8')

            # 计算图像在填充图像中的放置位置
            start_x = x_left_pad
            end_x = start_x + width
            start_y = y_up_pad
            end_y = start_y + height

            # 将图像放置在填充区域
            padded_image[start_y:end_y, start_x:end_x] = im
            image = padded_image
        else:
            image = im

        return image

