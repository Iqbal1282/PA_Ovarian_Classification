import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import StratifiedKFold
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt


img_size = 64 

train_transform_THB = A.Compose([
    # Resize with ratio range
	A.Resize(height=img_size, width=img_size, always_apply=True),
	A.ElasticTransform(alpha = 10, sigma = 250, p=0.5),
    A.GridDistortion(distort_limit=(-0.2,0.2), p=0.5),

    A.ShiftScaleRotate(shift_limit=(-0.005,0.005), scale_limit=(-0.2, 0.005), rotate_limit=(-30,30), border_mode=0, value=0, p=0.6),

	#A.RandomResizedCrop(size=(384, 384), scale=(0.9, 1.0), ratio=(0.9, 1.1), p=1.0),
	A.RandomResizedCrop(size=(img_size, img_size), scale=(0.8, 1.0), ratio=(0.75, 1.33), p=1.0),

    # Horizontal flip
    A.HorizontalFlip(p=0.5),

    # Normalize using paper settings (converted to single-channel equivalent)
    #A.Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=1.0),  # Adapted for grayscale
    A.Normalize(mean=(0.0002,), std=(0.0004,), max_pixel_value=0.0272),  # Adapted for grayscale

    # Ensure padding to crop size
    A.PadIfNeeded(min_height=384, min_width=384, border_mode=0, value=0, p=1.0),

    ToTensorV2()
])


val_transform_THB = A.Compose([
    A.Resize(img_size, img_size, p=1.0), 
    #A.Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=1.0), 
    A.Normalize(mean=(0.0002,), std=(0.0004,), max_pixel_value=0.0272),  # Adapted for grayscale
    ToTensorV2()
])


train_transform_SO2 = A.Compose([
    # Resize with ratio range
	A.Resize(height=img_size, width=img_size, always_apply=True),
	A.ElasticTransform(alpha = 10, sigma = 250, p=0.5),
    A.GridDistortion(distort_limit=(-0.2,0.2), p=0.5),

    A.ShiftScaleRotate(shift_limit=(-0.005,0.005), scale_limit=(-0.2, 0.005), rotate_limit=(-30,30), border_mode=0, value=0, p=0.6),

	#A.RandomResizedCrop(size=(384, 384), scale=(0.9, 1.0), ratio=(0.9, 1.1), p=1.0),
	A.RandomResizedCrop(size=(img_size, img_size), scale=(0.8, 1.0), ratio=(0.75, 1.33), p=1.0),

    # Horizontal flip
    A.HorizontalFlip(p=0.5),

    # Normalize using paper settings (converted to single-channel equivalent)
    A.Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=1.0),  # Adapted for grayscale
    #A.Normalize(mean=(0.0002,), std=(0.0004,), max_pixel_value=0.0272),  # Adapted for grayscale

    # Ensure padding to crop size
    A.PadIfNeeded(min_height=384, min_width=384, border_mode=0, value=0, p=1.0),

    ToTensorV2()
])


val_transform_SO2 = A.Compose([
    A.Resize(img_size, img_size, p=1.0), 
    A.Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=1.0), 
    #A.Normalize(mean=(0.0002,), std=(0.0004,), max_pixel_value=0.0272),  # Adapted for grayscale
    ToTensorV2()
])



class ROIMatDataset(Dataset):
    def __init__(self, 
                 csv_path, 
                 mat_root_dir, 
                 label_type='SO2', # SO2 or GT  #or Both 
                 phase='train',  # 'train', 'val', 'test'
                 k_fold=5, 
                 fold=0):

        self.phase = phase
        self.label_type = label_type
        self.root_dir = mat_root_dir

        assert self.label_type in ['Both', 'THb', 'SO2'], "Invalid label type!"

        # Load CSV
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=["PatientID", "Side", label_type])

        df["PatientID"] = df["PatientID"].astype(int)
        df["GT"] = (df["GT"] < 1).astype(int)
        df["PatientSide"] = df.apply(lambda row: f"p{row['PatientID']}_{row['Side']}", axis=1)

        # === Define test set ===
        df["IsTest"] = df["PatientID"] <= 60  # Customize as needed
        test_case_set = set(df[df["IsTest"]]["PatientSide"].unique())

        # Construct GT map per PatientSide
        grouped_gt = df.groupby("PatientSide")["GT"].agg(lambda x: x.mode()[0])
        case_ids = grouped_gt.index.tolist()
        case_labels = grouped_gt.values.tolist()
        label_map = grouped_gt.to_dict()

        # === Phase selection logic ===
        if phase == "test":
            selected_cases = test_case_set
            self.transform = val_transform_SO2 if label_type == 'SO2' else val_transform_THB
        else:
            strat_case_ids = [cid for cid in case_ids if cid not in test_case_set]
            strat_case_labels = [label_map[cid] for cid in strat_case_ids]

            skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=42)
            train_idx, val_idx = list(skf.split(strat_case_ids, strat_case_labels))[fold]

            if phase == "train":
                selected_cases = set(strat_case_ids[i] for i in train_idx)
                self.transform = train_transform_SO2 if label_type == 'SO2' else train_transform_THB
            else:
                selected_cases = set(strat_case_ids[i] for i in val_idx)
                self.transform = val_transform_SO2 if label_type == 'SO2' else val_transform_THB

        # === Filter and build dataset ===
        selected_df = df[df["PatientSide"].isin(selected_cases)]

        self.data = []
        for _, row in selected_df.iterrows():
            mat_path = os.path.join(mat_root_dir, row["Filename"])
            if not os.path.exists(mat_path):
                continue

            #label = row[label_type]
            gt_binary = int(row["GT"])  # use GT for stratification
            self.data.append({
                "mat_path": mat_path,
                #"label": label,
                "gt_binary": gt_binary
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        mat_contents = loadmat(item["mat_path"])
        img = mat_contents['img'].astype(np.float32)

        if img.ndim == 2:
            pass
        elif img.ndim == 3:
            img = np.transpose(img, (2, 0, 1))

        # Albumentations expects (H, W, C)
        if img.ndim == 2:
            img = np.expand_dims(img, axis=-1)
        else:
            img = np.transpose(img, (1, 2, 0))

        transformed = self.transform(image=img)
        img_tensor = transformed['image']

        label = torch.tensor([item["gt_binary"]], dtype=torch.float32)

        return img_tensor, label
    

# class PairedROIMatDataset(Dataset):
#     def __init__(self, 
#                  csv_path, 
#                  mat_root_dir, 
#                  phase='train',
#                  k_fold=5, 
#                  fold=0):

#         self.phase = phase
#         self.root_dir = mat_root_dir

#         df = pd.read_csv(csv_path)
#         df = df.dropna(subset=["PatientID", "Side", "SO2", "THb"])
#         df["PatientID"] = df["PatientID"].astype(int)
#         df["GT"] = (df["GT"] < 1).astype(int)
#         df["PatientSide"] = df.apply(lambda row: f"p{row['PatientID']}_{row['Side']}", axis=1)

#         df["IsTest"] = df["PatientID"] <= 60
#         test_case_set = set(df[df["IsTest"]]["PatientSide"].unique())

#         grouped_gt = df.groupby("PatientSide")["GT"].agg(lambda x: x.mode()[0])
#         case_ids = grouped_gt.index.tolist()
#         case_labels = grouped_gt.values.tolist()
#         label_map = grouped_gt.to_dict()

#         if phase == "test":
#             selected_cases = test_case_set
#             self.transform_so2 = val_transform_SO2
#             self.transform_thb = val_transform_THB
#         else:
#             strat_case_ids = [cid for cid in case_ids if cid not in test_case_set]
#             strat_case_labels = [label_map[cid] for cid in strat_case_ids]

#             skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=42)
#             train_idx, val_idx = list(skf.split(strat_case_ids, strat_case_labels))[fold]

#             if phase == "train":
#                 selected_cases = set(strat_case_ids[i] for i in train_idx)
#                 self.transform_so2 = train_transform_SO2
#                 self.transform_thb = train_transform_THB
#             else:
#                 selected_cases = set(strat_case_ids[i] for i in val_idx)
#                 self.transform_so2 = val_transform_SO2
#                 self.transform_thb = val_transform_THB

#         selected_df = df[df["PatientSide"].isin(selected_cases)]

#         self.data = []
#         grouped = selected_df.groupby("PatientSide")
#         for _, group in grouped:
#             so2_rows = group[group["Filename"].str.contains("SO2")]
#             thb_rows = group[group["Filename"].str.contains("THb")]
#             for _, so2_row in so2_rows.iterrows():
#                 for _, thb_row in thb_rows.iterrows():
#                     so2_path = os.path.join(mat_root_dir, so2_row["Filename"])
#                     thb_path = os.path.join(mat_root_dir, thb_row["Filename"])
#                     if os.path.exists(so2_path) and os.path.exists(thb_path):
#                         self.data.append({
#                             "so2_path": so2_path,
#                             "thb_path": thb_path,
#                             "gt_binary": int(so2_row["GT"])
#                         })

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         item = self.data[idx]

#         so2_img = loadmat(item["so2_path"])['img'].astype(np.float32)
#         thb_img = loadmat(item["thb_path"])['img'].astype(np.float32)

#         if so2_img.ndim == 2:
#             so2_img = np.expand_dims(so2_img, axis=-1)
#         else:
#             so2_img = np.transpose(so2_img, (1, 2, 0))

#         if thb_img.ndim == 2:
#             thb_img = np.expand_dims(thb_img, axis=-1)
#         else:
#             thb_img = np.transpose(thb_img, (1, 2, 0))

#         so2_tensor = self.transform_so2(image=so2_img)["image"]
#         thb_tensor = self.transform_thb(image=thb_img)["image"]
#         label = torch.tensor([item["gt_binary"]], dtype=torch.float32)

#         return so2_tensor, thb_tensor, label


class PairedROIMatDataset(Dataset):
    def __init__(self, 
                 so2_csv_path, 
                 thb_csv_path,
                 mat_root_dir, 
                 phase='train',
                 k_fold=5, 
                 fold=0):

        self.phase = phase
        self.root_dir = mat_root_dir

        # Load and process both SO2 and THb CSVs
        so2_df = pd.read_csv(so2_csv_path)
        thb_df = pd.read_csv(thb_csv_path)

        # Ensure necessary columns exist
        so2_df = so2_df.dropna(subset=["PatientID", "Side", "SO2"])
        thb_df = thb_df.dropna(subset=["PatientID", "Side", "THb"])

        # Standardize column types
        so2_df["PatientID"] = so2_df["PatientID"].astype(int)
        thb_df["PatientID"] = thb_df["PatientID"].astype(int)

        # Create PatientSide identifier
        so2_df["PatientSide"] = so2_df.apply(lambda row: f"p{row['PatientID']}_{row['Side']}", axis=1)
        thb_df["PatientSide"] = thb_df.apply(lambda row: f"p{row['PatientID']}_{row['Side']}", axis=1)

        # Merge SO2 and THb on PatientSide with all combinations
        merged_df = pd.merge(so2_df, thb_df, on="PatientSide", suffixes=("_so2", "_thb"))

        # Filter valid cases and assign labels
        merged_df = merged_df.dropna(subset=["Filename_so2", "Filename_thb", "GT_so2"])
        merged_df["GT"] = (merged_df["GT_so2"] < 1).astype(int)

        merged_df["IsTest"] = merged_df["PatientID_so2"] <= 60
        test_case_set = set(merged_df[merged_df["IsTest"]]["PatientSide"].unique())

        grouped_gt = merged_df.groupby("PatientSide")["GT"].agg(lambda x: x.mode()[0])
        case_ids = grouped_gt.index.tolist()
        case_labels = grouped_gt.values.tolist()
        label_map = grouped_gt.to_dict()

        if phase == "test":
            selected_cases = test_case_set
            self.transform_so2 = val_transform_SO2
            self.transform_thb = val_transform_THB
        else:
            strat_case_ids = [cid for cid in case_ids if cid not in test_case_set]
            strat_case_labels = [label_map[cid] for cid in strat_case_ids]

            skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=42)
            train_idx, val_idx = list(skf.split(strat_case_ids, strat_case_labels))[fold]

            if phase == "train":
                selected_cases = set(strat_case_ids[i] for i in train_idx)
                self.transform_so2 = train_transform_SO2
                self.transform_thb = train_transform_THB
            else:
                selected_cases = set(strat_case_ids[i] for i in val_idx)
                self.transform_so2 = val_transform_SO2
                self.transform_thb = val_transform_THB

        selected_df = merged_df[merged_df["PatientSide"].isin(selected_cases)]

        self.data = []
        for _, row in selected_df.iterrows():
            so2_path = os.path.join(mat_root_dir, row["Filename_so2"])
            thb_path = os.path.join(mat_root_dir, row["Filename_thb"])
            if os.path.exists(so2_path) and os.path.exists(thb_path):
                self.data.append({
                    "so2_path": so2_path,
                    "thb_path": thb_path,
                    "gt_binary": int(row["GT"])
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        so2_img = loadmat(item["so2_path"])['img'].astype(np.float32)
        thb_img = loadmat(item["thb_path"])['img'].astype(np.float32)

        if so2_img.ndim == 2:
            so2_img = np.expand_dims(so2_img, axis=-1)
        if thb_img.ndim == 2:
            thb_img = np.expand_dims(thb_img, axis=-1)

        so2_img = self.transform_so2(image=so2_img)['image']
        thb_img = self.transform_thb(image=thb_img)['image']

        label = torch.tensor([item["gt_binary"]], dtype=torch.float32)

        return so2_img, thb_img, label



if __name__ == "__main__":
    # from torch.utils.data import DataLoader

    # dataset = ROIMatDataset(
    #     csv_path='PAT features/roi_so2_image_metadata.csv',
    #     mat_root_dir='PAT features/ROI_MAT',
    #     label_type='SO2',
    #     phase='val',  # 'train', 'val', 'test'
    #     k_fold=5,
    #     fold=3
    # )

    # loader = DataLoader(dataset, batch_size=1, shuffle=True)


    from torch.utils.data import DataLoader

    dataset = PairedROIMatDataset(
        so2_csv_path ='PAT features/roi_so2_image_metadata.csv',
        thb_csv_path= 'PAT features/roi_thb_image_metadata.csv',
        mat_root_dir='PAT features/ROI_MAT',
        phase='train',
        k_fold=5,
        fold=0
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Sample batch
    for images, image2, labels in loader:
        print(images.shape)  # (B, C, H, W)
        print(labels.shape)  # (B,)
        plt.figure()
		#plt.subplot(1,3,1)
        plt.imshow(images[0][0], cmap= 'gray')
        plt.show()
        break


    Malignant_count = 0 
    Benign_count = 0 


    for images,images2, labels in loader:
        if labels == 0: 
            Benign_count+=1 
        else: 
            Malignant_count+=1
    print("Malignant: ", Malignant_count)
    print("Benign: ", Benign_count)
