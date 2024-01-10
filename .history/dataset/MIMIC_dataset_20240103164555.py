# import pandas as pd
# import collections
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from PIL import Image
# import os
# import cv2
# import torch
# import numpy as np
# import random
# import sys
# import warnings
# import zipfile
# import skimage
# from skimage.io import imread
# from typing import List, Dict
# from PIL import Image
# random.seed(42)
 
# def normalize(img, maxval, reshape=False):
#     """Scales images to be roughly [-1024 1024]."""
 
#     if img.max() > maxval:
#         raise Exception("max image value ({}) higher than expected bound ({}).".format(img.max(), maxval))
 
#     img = (2 * (img.astype(np.float32) / maxval) - 1.) * 1024
 
#     if reshape:
#         # Check that images are 2D arrays
#         if len(img.shape) > 2:
#             img = img[:, :, 0]
#         if len(img.shape) < 2:
#             print("error, dimension lower than 2 for image")
 
#         # add color channel
#         img = img[None, :, :]
#     return img
 
# def apply_transforms(sample, transform, seed=None) -> Dict:
#     """Applies transforms to the image and masks.
#     The seeds are set so that the transforms that are applied
#     to the image are the same that are applied to each mask.
#     This way data augmentation will work for segmentation or
#     other tasks which use masks information.
#     """
 
#     if seed is None:
#         MAX_RAND_VAL = 2147483647
#         seed = np.random.randint(MAX_RAND_VAL)
 
#     if transform is not None:
#         random.seed(seed)
#         torch.random.manual_seed(seed)
#         sample["img"] = transform(sample["img"])
 
#         if "pathology_masks" in sample:
#             for i in sample["pathology_masks"].keys():
#                 random.seed(seed)
#                 torch.random.manual_seed(seed)
#                 sample["pathology_masks"][i] = transform(sample["pathology_masks"][i])
 
#         if "semantic_masks" in sample:
#             for i in sample["semantic_masks"].keys():
#                 random.seed(seed)
#                 torch.random.manual_seed(seed)
#                 sample["semantic_masks"][i] = transform(sample["semantic_masks"][i])
 
#     return sample
 
# class Dataset:
#     """The datasets in this library aim to fit a simple interface where the
#     imgpath and csvpath are specified. Some datasets require more than one
#     metadata file and for some the metadata files are packaged in the library
#     so only the imgpath needs to be specified.
#     """
#     def __init__(self):
#         pass
 
#     pathologies: List[str]
#     """A list of strings identifying the pathologies contained in this
#     dataset. This list corresponds to the columns of the `.labels` matrix.
#     Although it is called pathologies, the contents do not have to be
#     pathologies and may simply be attributes of the patient. """
 
#     labels: np.ndarray
#     """A NumPy array which contains a 1, 0, or NaN for each pathology. Each
#     column is a pathology and each row corresponds to an item in the dataset.
#     A 1 represents that the pathology is present, 0 represents the pathology
#     is absent, and NaN represents no information. """
 
#     csv: pd.DataFrame
#     """A Pandas DataFrame of the metadata .csv file that is included with the
#     data. For some datasets multiple metadata files have been merged
#     together. It is largely a "catch-all" for associated data and the
#     referenced publication should explain each field. Each row aligns with
#     the elements of the dataset so indexing using .iloc will work. Alignment
#     between the DataFrame and the dataset items will be maintained when using
#     tools from this library. """
 
#     def totals(self) -> Dict[str, Dict[str, int]]:
#         """Compute counts of pathologies.
 
#         Returns: A dict containing pathology name -> (label->value)
#         """
#         counts = [dict(collections.Counter(items[~np.isnan(items)]).most_common()) for items in self.labels.T]
#         return dict(zip(self.pathologies, counts))
 
#     # def __repr__(self) -> str:
#     #     """Returns the name and a description of the dataset such as:
 
#     #     .. code-block:: python
 
#     #         CheX_Dataset num_samples=191010 views=['PA', 'AP']
 
#     #     If in a jupyter notebook it will also print the counts of the
#     #     pathology counts returned by .totals()
 
#     #     .. code-block:: python
 
#     #         {'Atelectasis': {0.0: 17621, 1.0: 29718},
#     #          'Cardiomegaly': {0.0: 22645, 1.0: 23384},
#     #          'Consolidation': {0.0: 30463, 1.0: 12982},
#     #          ...}
 
#     #     """
#     #     if xrv.utils.in_notebook():
#     #         pprint.pprint(self.totals())
#     #     return self.string()
 
#     def check_paths_exist(self):
#         if not os.path.isdir(self.imgpath):
#             raise Exception("imgpath must be a directory")
#         if not os.path.isfile(self.csvpath):
#             raise Exception("csvpath must be a file")
 
#     def limit_to_selected_views(self, views):
#         """This function is called by subclasses to filter the
#         images by view based on the values in .csv['view']
#         """
#         if type(views) is not list:
#             views = [views]
#         if '*' in views:
#             # if you have the wildcard, the rest are irrelevant
#             views = ["*"]
#         self.views = views
 
#         # missing data is unknown
#         self.csv.view.fillna("UNKNOWN", inplace=True)
 
#         if "*" not in views:
#             self.csv = self.csv[self.csv["view"].isin(self.views)]  # Select the view
 
# class MIMIC_Dataset(Dataset):
#     """MIMIC-CXR Dataset
 
#     Citation:
 
#     Johnson AE, Pollard TJ, Berkowitz S, Greenbaum NR, Lungren MP, Deng CY,
#     Mark RG, Horng S. MIMIC-CXR: A large publicly available database of
#     labeled chest radiographs. arXiv preprint arXiv:1901.07042. 2019 Jan 21.
 
#     https://arxiv.org/abs/1901.07042
 
#     Dataset website here:
#     https://physionet.org/content/mimic-cxr-jpg/2.0.0/
#     """
 
#     def __init__(self,
#                  imgpath,
#                  csvpath,
#                  metacsvpath,
#                  splitpath,
#                  views=["PA"],
#                  transform=None,
#                  data_aug=None,
#                  seed=0,
#                  split = 'train',
#                  unique_patients=True
#                  ):
 
#         super(MIMIC_Dataset, self).__init__()
#         np.random.seed(seed)  # Reset the seed so all runs are the same.
 
#         self.pathologies = ["Enlarged Cardiomediastinum",
#                             "Cardiomegaly",
#                             "Lung Opacity",
#                             "Lung Lesion",
#                             "Edema",
#                             "Consolidation",
#                             "Pneumonia",
#                             "Atelectasis",
#                             "Pneumothorax",
#                             "Pleural Effusion",
#                             "Pleural Other",
#                             "Fracture",
#                             "Support Devices"]
 
#         self.pathologies = sorted(self.pathologies)
 
#         self.imgpath = imgpath
#         self.transform = transform
#         self.data_aug = data_aug
#         self.csvpath = csvpath
#         self.csv = pd.read_csv(self.csvpath)
#         self.metacsvpath = metacsvpath
#         self.splitpath = splitpath
#         # self.PIL_transform = transforms.ToPILImage()
       
#         self.metacsv = pd.read_csv(self.metacsvpath)
#         # print('metaaaaaaaaaaaaaaa',self.metacsv)
#         self.split_dataset = pd.read_csv(self.splitpath)
#         test_df = self.split_dataset[(self.split_dataset['split'] == split)]
#         test_df.reset_index(drop=True, inplace=True)
#         # print('testttttttttttttttt',test_df)
 
#         final_df = pd.merge(test_df, self.metacsv, on=['dicom_id', 'subject_id', 'study_id'], how='inner')
#         final_df = final_df[self.metacsv.columns]
 
#         self.csv = self.csv.set_index(['subject_id', 'study_id'])
#         final_df = final_df.set_index(['subject_id', 'study_id'])
#         print('hiiiiiiiiiiiiiiiiiii')
#         self.csv = self.csv.join(final_df, how='inner').reset_index()
#         # Keep only the desired view
#         self.csv["view"] = self.csv["ViewPosition"]
#         self.limit_to_selected_views(views)
 
#         if unique_patients:
#             self.csv = self.csv.groupby("subject_id").first().reset_index()
#         print('helllllllllllooooooooooooo')
 
#         # Get our classes.
#         healthy = self.csv["No Finding"] == 1
#         labels = []
#         for pathology in self.pathologies:
#             if pathology in self.csv.columns:
#                 self.csv.loc[healthy, pathology] = 0
#                 mask = self.csv[pathology]
 
#             labels.append(mask.values)
#         print('byereeeeeeeeeeee')
 
#         self.labels = np.asarray(labels).T
#         self.labels = self.labels.astype(np.float32)
 
#         self.labels[self.labels == -1] = np.nan
#         # print(self.labels.shape)
#         self.pathologies = list(np.char.replace(self.pathologies, "Pleural Effusion", "Effusion"))
#         print('edfsdgfhgfrhtdfghgfd')
 
 
#         # offset_day_int
#         self.csv["offset_day_int"] = self.csv["StudyDate"]
 
#         # patientid
#         self.csv["patientid"] = self.csv["subject_id"].astype(str)
#         # print('final df', self.csv)
 
#     def string(self):
#         return self.__class__.__name__ + " num_samples={} views={} data_aug={}".format(len(self), self.views, self.data_aug)
 
#     def __len__(self):
#         return len(self.labels)
 
#     def __getitem__(self, idx):
#         sample = {}
#         sample["idx"] = idx
#         sample["lab"] = self.labels[idx]
 
#         subjectid = str(self.csv.iloc[idx]["subject_id"])
#         studyid = str(self.csv.iloc[idx]["study_id"])
#         dicom_id = str(self.csv.iloc[idx]["dicom_id"])
 
#         img_path = os.path.join(self.imgpath, "p" + subjectid[:2], "p" + subjectid, "s" + studyid, dicom_id + ".jpg")
#         # img_path = os.path.join(self.imgpath, dicom_id + '.jpg' + '_' + 'p' + subjectid[:2] + '_' + 'p' + subjectid + '_' + 's' + studyid + '_' + 'GT_img1' + '.jpeg')
#         # print(img_path)
#         img = imread(img_path)
#         # img = Image.fromarray(img)
#         img = np.expand_dims(img,axis=0)
#         # img = imread(img_path)
#         img = torch.from_numpy(img)
#         img = torch.cat([img, img, img], dim=0)
#         sample["img"] = img
#         print('beforeeeee', sample["img"].shape)
#         # img = img.detach().cpu().numpy()
#         # print(img.shape)
#         # sample["img"] = normalize(img, maxval=255, reshape=False)
#         # print(sample["img"], sample["img"].shape)
#         # sample = apply_transforms(sample, self.transform)
#         # sample['img'] = sample['img'].transpose(1,2,0)
#         sample = apply_transforms(sample, self.data_aug)
#         print(sample["img"].shape)
#         # print(sample["img"], sample["img"].shape)
#         return sample
 
 
 
# class XRayResizer(object):
#     """Resize an image to a specific size"""
#     def __init__(self, size: int, engine="skimage"):
#         self.size = size
#         self.engine = engine
#         if 'cv2' in sys.modules:
#             print("Setting XRayResizer engine to cv2 could increase performance.")
 
#     def __call__(self, img: np.ndarray) -> np.ndarray:
#         if self.engine == "skimage":
#             with warnings.catch_warnings():
#                 warnings.simplefilter("ignore")
#                 return skimage.transform.resize(img, (1, self.size, self.size), mode='constant', preserve_range=True).astype(np.float32)
#         elif self.engine == "cv2":
#             return cv2.resize(img[0, :, :],
#                               (self.size, self.size),
#                               interpolation=cv2.INTER_AREA
#                               ).reshape(1, self.size, self.size).astype(np.float32)
#         else:
#             raise Exception("Unknown engine, Must be skimage (default) or cv2.")
 
 
# class XRayCenterCrop(object):
#     """Perform a center crop on the long dimension of the input image"""
#     def crop_center(self, img: np.ndarray) -> np.ndarray:
#         _, y, x = img.shape
#         crop_size = np.min([y, x])
#         startx = x // 2 - (crop_size // 2)
#         starty = y // 2 - (crop_size // 2)
#         return img[:, starty:starty + crop_size, startx:startx + crop_size]
 
#     def __call__(self, img: np.ndarray) -> np.ndarray:
#         return self.crop_center(img)
    




# class MIMIC_Dataset_v2(Dataset):
#     """MIMIC-CXR Dataset
 
#     Citation:
 
#     Johnson AE, Pollard TJ, Berkowitz S, Greenbaum NR, Lungren MP, Deng CY,
#     Mark RG, Horng S. MIMIC-CXR: A large publicly available database of
#     labeled chest radiographs. arXiv preprint arXiv:1901.07042. 2019 Jan 21.
 
#     https://arxiv.org/abs/1901.07042
 
#     Dataset website here:
#     https://physionet.org/content/mimic-cxr-jpg/2.0.0/
#     """
 
#     def __init__(self,
#                  imgpath,
#                  csvpath,
#                  metacsvpath,
#                  splitpath,
#                  views=["PA"],
#                  transform=None,
#                  data_aug=None,
#                  seed=0,
#                  split = 'train',
#                  tag = '1_1',
#                  unique_patients=True,
#                  healthy = False,
#                  ):
 
#         super(MIMIC_Dataset_v2, self).__init__()
#         np.random.seed(seed)  # Reset the seed so all runs are the same.
 
#         self.pathologies = ["Enlarged Cardiomediastinum",
#                             "Cardiomegaly",
#                             "Lung Opacity",
#                             "Lung Lesion",
#                             "Edema",
#                             "Consolidation",
#                             "Pneumonia",
#                             "Atelectasis",
#                             "Pneumothorax",
#                             "Pleural Effusion",
#                             "Pleural Other",
#                             "Fracture",
#                             "Support Devices"]
 
#         self.pathologies = sorted(self.pathologies)
#         healthy_value = healthy
#         self.imgpath = imgpath
#         self.transform = transform
#         self.data_aug = data_aug
#         self.csvpath = csvpath
#         self.metacsvpath = metacsvpath
#         self.splitpath = splitpath
#         # self.PIL_transform = transforms.ToPILImage()
#         self.tag = tag
#         meta_header = ['dicom_id', 'subject_id', 'study_id', 'ViewPosition', 'StudyDate', 'StudyTime','total_num','study_id_count']
#         csv_header = ['dicom_id', 'subject_id', 'study_id', 'Atelectasis', 'Cardiomegaly', 'Consolidation','Edema','Enlarged Cardiomediastinum','Fracture','Lung Lesion','Lung Opacity','No Finding','Pleural Effusion','Pleural Other','Pneumonia','Pneumothorax','Support Devices','total_num','study_id_count']
        
#         self.metacsv = pd.read_csv(self.metacsvpath, names = meta_header)
#         self.csv = pd.read_csv(self.csvpath, names = csv_header)
#         if self.tag == '1_2':
#             filtered_rows = self.metacsv[(self.metacsv['total_num'] == 2) & (self.metacsv['study_id_count'] == 1)]
#             # filtered_rows = filtered_rows.drop_duplicates(subset=['study_id'])
#         elif self.tag == '0_2':
#             filtered_rows = self.metacsv[(self.metacsv['total_num'] == 2) & (self.metacsv['study_id_count'] == 0)]
        
#         elif self.tag == '1_1':
#             print('yesssssssss')
#             filtered_rows = self.metacsv[(self.metacsv['total_num'] == 1)]
#         elif self.tag == 'all':
#             filtered_rows = self.metacsv
#         elif self.tag == 'both':
#             filtered_rows = self.metacsv[((self.metacsv['total_num'] == 2) & (self.metacsv['study_id_count'] == 1)) | (self.metacsv['total_num'] == 1)]
#         print(filtered_rows)
#         # print('metaaaaaaaaaaaaaaa',self.metacsv)
#         # self.split_dataset = pd.read_csv(self.splitpath)
#         # test_df = self.split_dataset[(self.split_dataset['split'] == split)]
#         # test_df.reset_index(drop=True, inplace=True)
#         # # print('testttttttttttttttt',test_df)
 
#         filtered_rows_v2 = pd.merge(filtered_rows, self.csv, on=['dicom_id', 'subject_id', 'study_id','total_num','study_id_count'], how='inner')
#         # final_df = final_df[self.metacsv.columns]
 
#         # self.csv = self.csv.set_index(['subject_id', 'study_id'])
#         # filtered_rows_v2 = filtered_rows_v2.set_index(['subject_id', 'study_id'])
#         self.csv = filtered_rows_v2

#         # print('hiiiiiiiiiiiiiiiiiii')
#         # self.csv = self.csv.join(filtered_rows_v2, how='inner').reset_index()
#         # Keep only the desired view
#         # self.csv["view"] = self.csv["ViewPosition"]
#         # self.limit_to_selected_views(views)

#         if unique_patients:
#             self.csv = self.csv.groupby("subject_id").first().reset_index()
#         # print('helllllllllllooooooooooooo')
 
#         # Get our classes.
#         healthy = self.csv["No Finding"] == 1
#         if healthy_value == True:
#             healthy_rows = self.csv[healthy]
#             self.csv = healthy_rows

#         # print(len(healthy))
#         # breakpoint()
#         labels = []
#         for pathology in self.pathologies:
#             if pathology in self.csv.columns:
#                 self.csv.loc[healthy, pathology] = 0
#                 mask = self.csv[pathology]
 
#             labels.append(mask.values)
#         # print('byereeeeeeeeeeee')
 
#         self.labels = np.asarray(labels).T
#         self.labels = self.labels.astype(np.float32)
 
#         self.labels[self.labels == -1] = np.nan
#         # print(self.labels.shape)
#         self.pathologies = list(np.char.replace(self.pathologies, "Pleural Effusion", "Effusion"))
#         # print('edfsdgfhgfrhtdfghgfd')
 
 
#         # # offset_day_int
#         # self.csv["offset_day_int"] = self.csv["StudyDate"]
 
#         # patientid
#         self.csv["patientid"] = self.csv["subject_id"].astype(str)
        
#         print(self.csv)


#             # breakpoint()
#         # pathologies = self.csv.columns[8:-2].to_list()
#         # def labeling(row):
#         #     label = ""
#         #     for path in pathologies:
#         #         if row[path] ==1:
#         #             label = label+path
#         #     return label

#         # self.csv['image_name'] = self.csv.apply(lambda row: f"{row['dicom_id']}_{row['subject_id'].split('_')[0]}_{row['study_id']}.jpg", axis=1)
#         # # breakpoint()
#         # self.csv['labels'] = self.csv.apply(lambda row: f"{row['dicom_id']}_{row['subject_id'].split('_')[0]}_{row['study_id']}.jpg", axis=1)


#         print('final df', self.csv)
#     def string(self):
#         return self.__class__.__name__ + " num_samples={} views={} data_aug={}".format(len(self), self.views, self.data_aug)
 
#     def __len__(self):
#         return len(self.labels)
 
#     def __getitem__(self, idx):
#         sample = {}
#         sample["idx"] = idx
#         sample["lab"] = self.labels[idx]
 
#         subjectid = str(self.csv.iloc[idx]["subject_id"]).split('_')[0]
#         studyid = str(self.csv.iloc[idx]["study_id"])
#         dicom_id = str(self.csv.iloc[idx]["dicom_id"])
 
#         img_path = os.path.join(self.imgpath, "p" + subjectid[:2] + "_" + "p" + str(self.csv.iloc[idx]["subject_id"]) + '_' +  "s" + studyid + '_' + dicom_id + ".jpg")
#         # img_path = os.path.join(self.imgpath, dicom_id + '.jpg')
#         # img_path = os.path.join(self.imgpath, 'p' + subjectid[:2] + '_' + str(self.csv.iloc[idx]["subject_id"]) + '_' +'p'+subjectid + '_' + 's' + studyid + '_gen_img1.jpeg')
#         # img_path = os.path.join(self.imgpath, dicom_id + '.jpg' + '_' + 'p' + subjectid[:2] + '_' + 'p' + subjectid + '_' + 's' + studyid + '_' + 'GT_img1' + '.jpeg')
#         # img_path = os.path.join(self.imgpath, dicom_id + '_' +  str(self.csv.iloc[idx]["subject_id"]) + '_' + studyid + '.jpg')
#         # print(img_path)
#         # print(img_path)
#         img_path = os.path.join(self.imgpath, subjectid + "_" + studyid + '_' + dicom_id + ".png")

#         img = imread(img_path)
#         img = img[:, :, :3]
#         # print(img.shape)
#         # img = Image.fromarray(img)
#         # img = np.expand_dims(img,axis=0)
#         # img = imread(img_path)
#         img = torch.from_numpy(img)
#         # img = torch.unsqueeze(img,2)
#         # img = torch.cat([img, img, img], dim=0)
#         sample["img"] = img
#         # print('beforeeeee', sample["img"].shape)
#         # img = img.detach().cpu().numpy()
#         # print(sample["img"].shape)
#         # sample["img"] = normalize(img, maxval=255, reshape=False)
#         # print(sample["img"], sample["img"].shape)
#         sample = apply_transforms(sample, self.transform)
#         sample['img'] = sample['img'].permute(2,0,1)
#         sample = apply_transforms(sample, self.data_aug)
#         # print(sample["img"].shape)
#         # print(sample["img"], sample["img"].shape)
#         return sample
 
 
import pandas as pd
import collections
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import cv2
import torch
import numpy as np
import random
import sys
import warnings
import zipfile
import skimage
from skimage.io import imread
from typing import List, Dict
from PIL import Image
random.seed(42)
 
def normalize(img, maxval, reshape=False):
    """Scales images to be roughly [-1024 1024]."""
 
    if img.max() > maxval:
        raise Exception("max image value ({}) higher than expected bound ({}).".format(img.max(), maxval))
 
    img = (2 * (img.astype(np.float32) / maxval) - 1.) * 1024
 
    if reshape:
        # Check that images are 2D arrays
        if len(img.shape) > 2:
            img = img[:, :, 0]
        if len(img.shape) < 2:
            print("error, dimension lower than 2 for image")
 
        # add color channel
        img = img[None, :, :]
    return img
 
def apply_transforms(sample, transform, seed=None) -> Dict:
    """Applies transforms to the image and masks.
    The seeds are set so that the transforms that are applied
    to the image are the same that are applied to each mask.
    This way data augmentation will work for segmentation or
    other tasks which use masks information.
    """
 
    if seed is None:
        MAX_RAND_VAL = 2147483647
        seed = np.random.randint(MAX_RAND_VAL)
 
    if transform is not None:
        random.seed(seed)
        torch.random.manual_seed(seed)
        sample["img"] = transform(sample["img"])
 
        if "pathology_masks" in sample:
            for i in sample["pathology_masks"].keys():
                random.seed(seed)
                torch.random.manual_seed(seed)
                sample["pathology_masks"][i] = transform(sample["pathology_masks"][i])
 
        if "semantic_masks" in sample:
            for i in sample["semantic_masks"].keys():
                random.seed(seed)
                torch.random.manual_seed(seed)
                sample["semantic_masks"][i] = transform(sample["semantic_masks"][i])
 
    return sample
 
class Dataset:
    """The datasets in this library aim to fit a simple interface where the
    imgpath and csvpath are specified. Some datasets require more than one
    metadata file and for some the metadata files are packaged in the library
    so only the imgpath needs to be specified.
    """
    def __init__(self):
        pass
 
    pathologies: List[str]
    """A list of strings identifying the pathologies contained in this
    dataset. This list corresponds to the columns of the `.labels` matrix.
    Although it is called pathologies, the contents do not have to be
    pathologies and may simply be attributes of the patient. """
 
    labels: np.ndarray
    """A NumPy array which contains a 1, 0, or NaN for each pathology. Each
    column is a pathology and each row corresponds to an item in the dataset.
    A 1 represents that the pathology is present, 0 represents the pathology
    is absent, and NaN represents no information. """
 
    csv: pd.DataFrame
    """A Pandas DataFrame of the metadata .csv file that is included with the
    data. For some datasets multiple metadata files have been merged
    together. It is largely a "catch-all" for associated data and the
    referenced publication should explain each field. Each row aligns with
    the elements of the dataset so indexing using .iloc will work. Alignment
    between the DataFrame and the dataset items will be maintained when using
    tools from this library. """
 
    def totals(self) -> Dict[str, Dict[str, int]]:
        """Compute counts of pathologies.
 
        Returns: A dict containing pathology name -> (label->value)
        """
        counts = [dict(collections.Counter(items[~np.isnan(items)]).most_common()) for items in self.labels.T]
        return dict(zip(self.pathologies, counts))
 
    # def __repr__(self) -> str:
    #     """Returns the name and a description of the dataset such as:
 
    #     .. code-block:: python
 
    #         CheX_Dataset num_samples=191010 views=['PA', 'AP']
 
    #     If in a jupyter notebook it will also print the counts of the
    #     pathology counts returned by .totals()
 
    #     .. code-block:: python
 
    #         {'Atelectasis': {0.0: 17621, 1.0: 29718},
    #          'Cardiomegaly': {0.0: 22645, 1.0: 23384},
    #          'Consolidation': {0.0: 30463, 1.0: 12982},
    #          ...}
 
    #     """
    #     if xrv.utils.in_notebook():
    #         pprint.pprint(self.totals())
    #     return self.string()
 
    def check_paths_exist(self):
        if not os.path.isdir(self.imgpath):
            raise Exception("imgpath must be a directory")
        if not os.path.isfile(self.csvpath):
            raise Exception("csvpath must be a file")
 
    def limit_to_selected_views(self, views):
        """This function is called by subclasses to filter the
        images by view based on the values in .csv['view']
        """
        if type(views) is not list:
            views = [views]
        if '*' in views:
            # if you have the wildcard, the rest are irrelevant
            views = ["*"]
        self.views = views
 
        # missing data is unknown
        self.csv.view.fillna("UNKNOWN", inplace=True)
 
        if "*" not in views:
            self.csv = self.csv[self.csv["view"].isin(self.views)]  # Select the view
 
class MIMIC_Dataset(Dataset):
    """MIMIC-CXR Dataset
 
    Citation:
 
    Johnson AE, Pollard TJ, Berkowitz S, Greenbaum NR, Lungren MP, Deng CY,
    Mark RG, Horng S. MIMIC-CXR: A large publicly available database of
    labeled chest radiographs. arXiv preprint arXiv:1901.07042. 2019 Jan 21.
 
    https://arxiv.org/abs/1901.07042
 
    Dataset website here:
    https://physionet.org/content/mimic-cxr-jpg/2.0.0/
    """
 
    def __init__(self,
                 imgpath,
                 csvpath,
                 metacsvpath,
                 splitpath,
                 views=["PA"],
                 transform=None,
                 data_aug=None,
                 seed=0,
                 split = 'train',
                 unique_patients=True
                 ):
 
        super(MIMIC_Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.
 
        self.pathologies = ["Enlarged Cardiomediastinum",
                            "Cardiomegaly",
                            "Lung Opacity",
                            "Lung Lesion",
                            "Edema",
                            "Consolidation",
                            "Pneumonia",
                            "Atelectasis",
                            "Pneumothorax",
                            "Pleural Effusion",
                            "Pleural Other",
                            "Fracture",
                            "Support Devices"]
 
        self.pathologies = sorted(self.pathologies)
 
        self.imgpath = imgpath
        self.transform = transform
        self.data_aug = data_aug
        self.csvpath = csvpath
        self.csv = pd.read_csv(self.csvpath)
        self.metacsvpath = metacsvpath
        self.splitpath = splitpath
        # self.PIL_transform = transforms.ToPILImage()
       
        self.metacsv = pd.read_csv(self.metacsvpath)
        # print('metaaaaaaaaaaaaaaa',self.metacsv)
        self.split_dataset = pd.read_csv(self.splitpath)
        test_df = self.split_dataset[(self.split_dataset['split'] == split)]
        test_df.reset_index(drop=True, inplace=True)
        # print('testttttttttttttttt',test_df)
 
        final_df = pd.merge(test_df, self.metacsv, on=['dicom_id', 'subject_id', 'study_id'], how='inner')
        final_df = final_df[self.metacsv.columns]
 
        self.csv = self.csv.set_index(['subject_id', 'study_id'])
        final_df = final_df.set_index(['subject_id', 'study_id'])
        print('hiiiiiiiiiiiiiiiiiii')
        self.csv = self.csv.join(final_df, how='inner').reset_index()
        # Keep only the desired view
        self.csv["view"] = self.csv["ViewPosition"]
        self.limit_to_selected_views(views)
 
        if unique_patients:
            self.csv = self.csv.groupby("subject_id").first().reset_index()
        print('helllllllllllooooooooooooo')
 
        # Get our classes.
        healthy = self.csv["No Finding"] == 1
        labels = []
        for pathology in self.pathologies:
            if pathology in self.csv.columns:
                self.csv.loc[healthy, pathology] = 0
                mask = self.csv[pathology]
 
            labels.append(mask.values)
        print('byereeeeeeeeeeee')
 
        self.labels = np.asarray(labels).T
        self.labels = self.labels.astype(np.float32)
 
        self.labels[self.labels == -1] = np.nan
        # print(self.labels.shape)
        self.pathologies = list(np.char.replace(self.pathologies, "Pleural Effusion", "Effusion"))
        print('edfsdgfhgfrhtdfghgfd')
 
 
        # offset_day_int
        self.csv["offset_day_int"] = self.csv["StudyDate"]
 
        # patientid
        self.csv["patientid"] = self.csv["subject_id"].astype(str)
        # print('final df', self.csv)
 
    def string(self):
        return self.__class__.__name__ + " num_samples={} views={} data_aug={}".format(len(self), self.views, self.data_aug)
 
    def __len__(self):
        return len(self.labels)
 
    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]
 
        subjectid = str(self.csv.iloc[idx]["subject_id"])
        studyid = str(self.csv.iloc[idx]["study_id"])
        dicom_id = str(self.csv.iloc[idx]["dicom_id"])
 
        img_path = os.path.join(self.imgpath, "p" + subjectid[:2], "p" + subjectid, "s" + studyid, dicom_id + ".jpg")
        # img_path = os.path.join(self.imgpath, dicom_id + '.jpg' + '_' + 'p' + subjectid[:2] + '_' + 'p' + subjectid + '_' + 's' + studyid + '_' + 'GT_img1' + '.jpeg')
        # print(img_path)
        img = imread(img_path)
        # img = Image.fromarray(img)
        img = np.expand_dims(img,axis=0)
        # img = imread(img_path)
        img = torch.from_numpy(img)
        img = torch.cat([img, img, img], dim=0)
        sample["img"] = img
        print('beforeeeee', sample["img"].shape)
        # img = img.detach().cpu().numpy()
        # print(img.shape)
        # sample["img"] = normalize(img, maxval=255, reshape=False)
        # print(sample["img"], sample["img"].shape)
        # sample = apply_transforms(sample, self.transform)
        # sample['img'] = sample['img'].transpose(1,2,0)
        sample = apply_transforms(sample, self.data_aug)
        print(sample["img"].shape)
        # print(sample["img"], sample["img"].shape)
        return sample
 
 
 
class XRayResizer(object):
    """Resize an image to a specific size"""
    def __init__(self, size: int, engine="skimage"):
        self.size = size
        self.engine = engine
        if 'cv2' in sys.modules:
            print("Setting XRayResizer engine to cv2 could increase performance.")
 
    def __call__(self, img: np.ndarray) -> np.ndarray:
        if self.engine == "skimage":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return skimage.transform.resize(img, (1, self.size, self.size), mode='constant', preserve_range=True).astype(np.float32)
        elif self.engine == "cv2":
            return cv2.resize(img[0, :, :],
                              (self.size, self.size),
                              interpolation=cv2.INTER_AREA
                              ).reshape(1, self.size, self.size).astype(np.float32)
        else:
            raise Exception("Unknown engine, Must be skimage (default) or cv2.")
 
 
class XRayCenterCrop(object):
    """Perform a center crop on the long dimension of the input image"""
    def crop_center(self, img: np.ndarray) -> np.ndarray:
        _, y, x = img.shape
        crop_size = np.min([y, x])
        startx = x // 2 - (crop_size // 2)
        starty = y // 2 - (crop_size // 2)
        return img[:, starty:starty + crop_size, startx:startx + crop_size]
 
    def __call__(self, img: np.ndarray) -> np.ndarray:
        return self.crop_center(img)
    




class MIMIC_Dataset_v2(Dataset):
    """MIMIC-CXR Dataset
 
    Citation:
 
    Johnson AE, Pollard TJ, Berkowitz S, Greenbaum NR, Lungren MP, Deng CY,
    Mark RG, Horng S. MIMIC-CXR: A large publicly available database of
    labeled chest radiographs. arXiv preprint arXiv:1901.07042. 2019 Jan 21.
 
    https://arxiv.org/abs/1901.07042
 
    Dataset website here:
    https://physionet.org/content/mimic-cxr-jpg/2.0.0/
    """
 
    def __init__(self,
                 imgpath,
                 csvpath,
                 metacsvpath,
                 splitpath,
                 views=["PA"],
                 transform=None,
                 data_aug=None,
                 seed=0,
                 split = 'train',
                 tag = '1_1',
                 unique_patients=True,
                 healthy = False,
                 ):
 
        super(MIMIC_Dataset_v2, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.
 
        self.pathologies = ["Enlarged Cardiomediastinum",
                            "Cardiomegaly",
                            "Lung Opacity",
                            "Lung Lesion",
                            "Edema",
                            "Consolidation",
                            "Pneumonia",
                            "Atelectasis",
                            "Pneumothorax",
                            "Pleural Effusion",
                            "Pleural Other",
                            "Fracture",
                            "Support Devices"]
 
        self.pathologies = sorted(self.pathologies)
        healthy_value = healthy
        self.imgpath = imgpath
        self.transform = transform
        self.data_aug = data_aug
        self.csvpath = csvpath
        self.metacsvpath = metacsvpath
        self.splitpath = splitpath
        # self.PIL_transform = transforms.ToPILImage()
        self.tag = tag
        meta_header = ['dicom_id', 'subject_id', 'study_id', 'ViewPosition', 'StudyDate', 'StudyTime','total_num','study_id_count']
        csv_header = ['dicom_id', 'subject_id', 'study_id', 'Atelectasis', 'Cardiomegaly', 'Consolidation','Edema','Enlarged Cardiomediastinum','Fracture','Lung Lesion','Lung Opacity','No Finding','Pleural Effusion','Pleural Other','Pneumonia','Pneumothorax','Support Devices','total_num','study_id_count']
        
        self.metacsv = pd.read_csv(self.metacsvpath, names = meta_header)
        self.csv = pd.read_csv(self.csvpath, names = csv_header)
        if self.tag == '1_2':
            filtered_rows = self.metacsv[(self.metacsv['total_num'] == 2) & (self.metacsv['study_id_count'] == 1)]
            # filtered_rows = filtered_rows.drop_duplicates(subset=['study_id'])
        elif self.tag == '0_2':
            filtered_rows = self.metacsv[(self.metacsv['total_num'] == 2) & (self.metacsv['study_id_count'] == 0)]
        
        elif self.tag == '1_1':
            print('yesssssssss')
            filtered_rows = self.metacsv[(self.metacsv['total_num'] == 1)]
        elif self.tag == 'all':
            filtered_rows = self.metacsv
        elif self.tag == 'both':
            filtered_rows = self.metacsv[((self.metacsv['total_num'] == 2) & (self.metacsv['study_id_count'] == 1)) | (self.metacsv['total_num'] == 1)]
        print(filtered_rows)
        # print('metaaaaaaaaaaaaaaa',self.metacsv)
        # self.split_dataset = pd.read_csv(self.splitpath)
        # test_df = self.split_dataset[(self.split_dataset['split'] == split)]
        # test_df.reset_index(drop=True, inplace=True)
        # # print('testttttttttttttttt',test_df)
 
        filtered_rows_v2 = pd.merge(filtered_rows, self.csv, on=['dicom_id', 'subject_id', 'study_id','total_num','study_id_count'], how='inner')
        # final_df = final_df[self.metacsv.columns]
 
        # self.csv = self.csv.set_index(['subject_id', 'study_id'])
        # filtered_rows_v2 = filtered_rows_v2.set_index(['subject_id', 'study_id'])
        self.csv = filtered_rows_v2

        # print('hiiiiiiiiiiiiiiiiiii')
        # self.csv = self.csv.join(filtered_rows_v2, how='inner').reset_index()
        # Keep only the desired view
        # self.csv["view"] = self.csv["ViewPosition"]
        # self.limit_to_selected_views(views)

        if unique_patients:
            self.csv = self.csv.groupby("subject_id").first().reset_index()
        # print('helllllllllllooooooooooooo')
 
        # Get our classes.
        healthy = self.csv["No Finding"] == 1
        if healthy_value == True:
            healthy_rows = self.csv[healthy]
            self.csv = healthy_rows

        # print(len(healthy))
        # breakpoint()
        labels = []
        for pathology in self.pathologies:
            if pathology in self.csv.columns:
                self.csv.loc[healthy, pathology] = 0
                mask = self.csv[pathology]
 
            labels.append(mask.values)
        # print('byereeeeeeeeeeee')
 
        self.labels = np.asarray(labels).T
        self.labels = self.labels.astype(np.float32)
 
        self.labels[self.labels == -1] = np.nan
        # print(self.labels.shape)
        self.pathologies = list(np.char.replace(self.pathologies, "Pleural Effusion", "Effusion"))
        # print('edfsdgfhgfrhtdfghgfd')
 
 
        # # offset_day_int
        # self.csv["offset_day_int"] = self.csv["StudyDate"]
 
        # patientid
        self.csv["patientid"] = self.csv["subject_id"].astype(str)
        
        print(self.csv)


            # breakpoint()
        # pathologies = self.csv.columns[8:-2].to_list()
        # def labeling(row):
        #     label = ""
        #     for path in pathologies:
        #         if row[path] ==1:
        #             label = label+path
        #     return label

        # self.csv['image_name'] = self.csv.apply(lambda row: f"{row['dicom_id']}_{row['subject_id'].split('_')[0]}_{row['study_id']}.jpg", axis=1)
        # # breakpoint()
        # self.csv['labels'] = self.csv.apply(lambda row: f"{row['dicom_id']}_{row['subject_id'].split('_')[0]}_{row['study_id']}.jpg", axis=1)


        print('final df', self.csv)
    def string(self):
        return self.__class__.__name__ + " num_samples={} views={} data_aug={}".format(len(self), self.views, self.data_aug)
 
    def __len__(self):
        return len(self.labels)
 
    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]
 
        subjectid = str(self.csv.iloc[idx]["subject_id"]).split('_')[0]
        studyid = str(self.csv.iloc[idx]["study_id"])
        dicom_id = str(self.csv.iloc[idx]["dicom_id"])
        #  /home/santosh.sanjeev/san_data_v2/physionet.org/files/mimic-cxr-jpg/2.0.0/files/p10/p10000032/s53911762/
        # img_path = os.path.join(self.imgpath, 'p' + subjectid[:2], 'p' + subjectid,'s' + studyid, dicom_id + '.jpg')
        # img_path = os.path.join(self.imgpath, "p" + subjectid[:2] + "_" + "p" + str(self.csv.iloc[idx]["subject_id"]) + '_' +  "s" + studyid + '_' + dicom_id + ".jpg")
        # img_path = os.path.join(self.imgpath, dicom_id + '.jpg')
        # img_path = os.path.join(self.imgpath, 'p' + subjectid[:2] + '_' + str(self.csv.iloc[idx]["subject_id"]) + '_' +'p'+subjectid + '_' + 's' + studyid + '_gen_img1.jpeg')
        # img_path = os.path.join(self.imgpath, dicom_id + '.jpg' + '_' + 'p' + subjectid[:2] + '_' + 'p' + subjectid + '_' + 's' + studyid + '_' + 'GT_img1' + '.jpeg')
        # img_path = os.path.join(self.imgpath, dicom_id + '_' +  str(self.csv.iloc[idx]["subject_id"]) + '_' + studyid + '.jpg')
        # print(img_path)
        # print(img_path)
        img_path = os.path.join(self.imgpath, subjectid + "_" + studyid + '_' + dicom_id + ".png")

        img = imread(img_path)
        img = img[:, :, :3]
        # print(img.shape)
        # img = Image.fromarray(img)
        # img = np.expand_dims(img,axis=0)
        # img = imread(img_path)
        img = torch.from_numpy(img)
        # img = torch.unsqueeze(img,0)
        # img = torch.cat([img, img, img], dim=0)
        sample["img"] = img
        # print('beforeeeee', sample["img"].shape)
        # img = img.detach().cpu().numpy()
        # print(sample["img"].shape)
        # sample["img"] = normalize(img, maxval=255, reshape=False)
        # print(sample["img"], sample["img"].shape)
        # sample = apply_transforms(sample, self.transform)
        print(sample["img"].shape)

        sample['img'] = sample['img'].permute(2,0,1)
        sample = apply_transforms(sample, self.data_aug)
        # print('sdsfsdvsdsdvdsv',sample["img"].shape, sample["img"])

        # print(sample["img"].shape)
        # print(sample["img"], sample["img"].shape)
        return sample
 
 

class MIMIC_Dataset_v3(Dataset):
    """MIMIC-CXR Dataset
 
    Citation:
 
    Johnson AE, Pollard TJ, Berkowitz S, Greenbaum NR, Lungren MP, Deng CY,
    Mark RG, Horng S. MIMIC-CXR: A large publicly available database of
    labeled chest radiographs. arXiv preprint arXiv:1901.07042. 2019 Jan 21.
 
    https://arxiv.org/abs/1901.07042
 
    Dataset website here:
    https://physionet.org/content/mimic-cxr-jpg/2.0.0/
    """
 
    def __init__(self,
                 imgpath,
                 csvpath,
                 metacsvpath,
                 splitpath,
                 views=["PA"],
                 transform=None,
                 data_aug=None,
                 seed=0,
                 split = 'train',
                 tag = '1_1',
                 unique_patients=True,
                 healthy = False,
                 ):
 
        super(MIMIC_Dataset_v3, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.
 
        self.pathologies = ["Enlarged Cardiomediastinum",
                            "Cardiomegaly",
                            "Lung Opacity",
                            "Lung Lesion",
                            "Edema",
                            "Consolidation",
                            "Pneumonia",
                            "Atelectasis",
                            "Pneumothorax",
                            "Pleural Effusion",
                            "Pleural Other",
                            "Fracture",
                            "Support Devices"]
 
        self.pathologies = sorted(self.pathologies)
        healthy_value = healthy
        self.imgpath = imgpath
        self.transform = transform
        self.data_aug = data_aug
        self.csvpath = csvpath
        self.metacsvpath = metacsvpath
        self.splitpath = splitpath
        # self.PIL_transform = transforms.ToPILImage()
        self.tag = tag
        meta_header = ['dicom_id', 'subject_id', 'study_id', 'ViewPosition', 'StudyDate', 'StudyTime','total_num','study_id_count']
        csv_header = ['dicom_id', 'subject_id', 'study_id', 'Atelectasis', 'Cardiomegaly', 'Consolidation','Edema','Enlarged Cardiomediastinum','Fracture','Lung Lesion','Lung Opacity','No Finding','Pleural Effusion','Pleural Other','Pneumonia','Pneumothorax','Support Devices','total_num','study_id_count']
        
        self.metacsv = pd.read_csv(self.metacsvpath, names = meta_header)
        self.csv = pd.read_csv(self.csvpath, names = csv_header)
        if self.tag == '1_2':
            filtered_rows = self.metacsv[(self.metacsv['total_num'] == 2) & (self.metacsv['study_id_count'] == 1)]
            # filtered_rows = filtered_rows.drop_duplicates(subset=['study_id'])
        elif self.tag == '0_2':
            filtered_rows = self.metacsv[(self.metacsv['total_num'] == 2) & (self.metacsv['study_id_count'] == 0)]
        
        elif self.tag == '1_1':
            print('yesssssssss')
            filtered_rows = self.metacsv[(self.metacsv['total_num'] == 1)]
        elif self.tag == 'all':
            filtered_rows = self.metacsv
        elif self.tag == 'both':
            filtered_rows = self.metacsv[((self.metacsv['total_num'] == 2) & (self.metacsv['study_id_count'] == 1)) | (self.metacsv['total_num'] == 1)]
        print(filtered_rows)
        # print('metaaaaaaaaaaaaaaa',self.metacsv)
        # self.split_dataset = pd.read_csv(self.splitpath)
        # test_df = self.split_dataset[(self.split_dataset['split'] == split)]
        # test_df.reset_index(drop=True, inplace=True)
        # # print('testttttttttttttttt',test_df)
 
        filtered_rows_v2 = pd.merge(filtered_rows, self.csv, on=['dicom_id', 'subject_id', 'study_id','total_num','study_id_count'], how='inner')
        # final_df = final_df[self.metacsv.columns]
 
        # self.csv = self.csv.set_index(['subject_id', 'study_id'])
        # filtered_rows_v2 = filtered_rows_v2.set_index(['subject_id', 'study_id'])
        self.csv = filtered_rows_v2

        # print('hiiiiiiiiiiiiiiiiiii')
        # self.csv = self.csv.join(filtered_rows_v2, how='inner').reset_index()
        # Keep only the desired view
        # self.csv["view"] = self.csv["ViewPosition"]
        # self.limit_to_selected_views(views)

        if unique_patients:
            self.csv = self.csv.groupby("subject_id").first().reset_index()
        # print('helllllllllllooooooooooooo')
 
        # Get our classes.
        healthy = self.csv["No Finding"] == 1
        if healthy_value == True:
            healthy_rows = self.csv[healthy]
            self.csv = healthy_rows
        # print('debuggggg',self.csv)
        # print(len(healthy))
        # breakpoint()
        labels = []
        for pathology in self.pathologies:
            if pathology in self.csv.columns:
                self.csv.loc[healthy, pathology] = 0
                mask = self.csv[pathology]
 
            labels.append(mask.values)
        # print('byereeeeeeeeeeee')
 
        self.labels = np.asarray(labels).T
        self.labels = self.labels.astype(np.float32)
 
        self.labels[self.labels == -1] = np.nan
        # print(self.labels.shape)
        self.pathologies = list(np.char.replace(self.pathologies, "Pleural Effusion", "Effusion"))
        # print('edfsdgfhgfrhtdfghgfd')
 
 
        # # offset_day_int
        # self.csv["offset_day_int"] = self.csv["StudyDate"]
 
        # patientid
        # self.csv["patientid"] = self.csv["subject_id"].astype(str)
        
        print(self.csv)


            # breakpoint()
        # pathologies = self.csv.columns[8:-2].to_list()
        # def labeling(row):
        #     label = ""
        #     for path in pathologies:
        #         if row[path] ==1:
        #             label = label+path
        #     return label

        # self.csv['image_name'] = self.csv.apply(lambda row: f"{row['dicom_id']}_{row['subject_id'].split('_')[0]}_{row['study_id']}.jpg", axis=1)
        # # breakpoint()
        # self.csv['labels'] = self.csv.apply(lambda row: f"{row['dicom_id']}_{row['subject_id'].split('_')[0]}_{row['study_id']}.jpg", axis=1)

        print(self.csv.columns.tolist())
        self.csv = self.csv.drop(columns = ['ViewPosition','StudyDate','StudyTime','No Finding','total_num','study_id_count'])
        print('final df', self.csv)

        self.pred_df = pd.read_csv('/home/santosh.sanjeev/bi_report_2_2_test.csv')
        self.pred_df = self.pred_df.drop(columns = ['Unnamed: 0'])
        self.pred_df.columns = self.csv.columns.tolist()
        #put the header of the other csv
        print('predddddddd',self.pred_df)
        self.preds = self.pred_df.iloc[:,-13:].values
        print(self.preds)

        






    def string(self):
        return self.__class__.__name__ + " num_samples={} views={} data_aug={}".format(len(self), self.views, self.data_aug)
 
    def __len__(self):
        return len(self.labels)
 
    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]
        sample["pred"] = self.preds[idx]
        # print('lab',sample['lab'])
        # print('pred',sample['pred'])
        # subjectid = str(self.csv.iloc[idx]["subject_id"]).split('_')[0]
        # studyid = str(self.csv.iloc[idx]["study_id"])
        # dicom_id = str(self.csv.iloc[idx]["dicom_id"])
 
        # img_path = os.path.join(self.imgpath, "p" + subjectid[:2] + "_" + "p" + str(self.csv.iloc[idx]["subject_id"]) + '_' +  "s" + studyid + '_' + dicom_id + ".jpg")
        # # img_path = os.path.join(self.imgpath, dicom_id + '.jpg')
        # # img_path = os.path.join(self.imgpath, 'p' + subjectid[:2] + '_' + str(self.csv.iloc[idx]["subject_id"]) + '_' +'p'+subjectid + '_' + 's' + studyid + '_gen_img1.jpeg')
        # # img_path = os.path.join(self.imgpath, dicom_id + '.jpg' + '_' + 'p' + subjectid[:2] + '_' + 'p' + subjectid + '_' + 's' + studyid + '_' + 'GT_img1' + '.jpeg')
        # # img_path = os.path.join(self.imgpath, dicom_id + '_' +  str(self.csv.iloc[idx]["subject_id"]) + '_' + studyid + '.jpg')
        # # print(img_path)
        # # print(img_path)
        # img_path = os.path.join(self.imgpath, subjectid + "_" + studyid + '_' + dicom_id + ".png")

        # img = imread(img_path)
        # img = img[:, :, :3]
        # # print(img.shape)
        # # img = Image.fromarray(img)
        # # img = np.expand_dims(img,axis=0)
        # # img = imread(img_path)
        # img = torch.from_numpy(img)
        # # img = torch.unsqueeze(img,2)
        # # img = torch.cat([img, img, img], dim=0)
        # sample["img"] = img
        # # print('beforeeeee', sample["img"].shape)
        # # img = img.detach().cpu().numpy()
        # print(sample["img"].shape)
        # # sample["img"] = normalize(img, maxval=255, reshape=False)
        # # print(sample["img"], sample["img"].shape)
        # sample = apply_transforms(sample, self.transform)
        # sample['img'] = sample['img'].permute(2,0,1)
        # sample = apply_transforms(sample, self.data_aug)
        # # print(sample["img"].shape)
        # # print(sample["img"], sample["img"].shape)
        return sample
 
 

 

class VinBrain_Dataset(Dataset):
    """VinBrain Dataset

    .. code-block:: python

        d_vin = xrv.datasets.VinBrain_Dataset(
            imgpath=".../train",
            csvpath=".../train.csv"
        )

    Nguyen, H. Q., Lam, K., Le, L. T., Pham, H. H., Tran, D. Q., Nguyen,
    D. B., Le, D. D., Pham, C. M., Tong, H. T. T., Dinh, D. H., Do, C. D.,
    Doan, L. T., Nguyen, C. N., Nguyen, B. T., Nguyen, Q. V., Hoang, A. D.,
    Phan, H. N., Nguyen, A. T., Ho, P. H., … Vu, V. (2020). VinDr-CXR: An
    open dataset of chest X-rays with radiologist’s annotations.
    http://arxiv.org/abs/2012.15029

    https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection
    """

    def __init__(self,
                 imgpath,
                 csvpath=None,
                 views=None,
                 transform=None,
                 data_aug=None,
                 seed=0,
                 pathology_masks=False
                 ):
        super(VinBrain_Dataset, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath

        # if csvpath == none:
        #     self.csvpath = os.path.join(datapath, "vinbigdata-train.csv.gz")
        # else:
        self.csvpath = csvpath

        self.transform = transform
        self.data_aug = data_aug
        self.pathology_masks = pathology_masks
        self.views = views

        self.pathologies = ['Aortic enlargement',
                            'Atelectasis',
                            'Calcification',
                            'Cardiomegaly',
                            'Consolidation',
                            'ILD',
                            'Infiltration',
                            'Lung Opacity',
                            'Nodule/Mass',
                            'Lesion',
                            'Effusion',
                            'Pleural_Thickening',
                            'Pneumothorax',
                            'Pulmonary Fibrosis']

        self.pathologies = sorted(np.unique(self.pathologies))

        self.mapping = dict()
        self.mapping["Pleural_Thickening"] = ["Pleural thickening"]
        self.mapping["Effusion"] = ["Pleural effusion"]

        # Load data
        self.check_paths_exist()
        self.rawcsv = pd.read_csv(self.csvpath)
        self.csv = pd.DataFrame(self.rawcsv.groupby("image_id")["class_name"].apply(lambda x: "|".join(np.unique(x))))

        self.csv["has_masks"] = self.csv.class_name != "No finding"

        labels = []
        for pathology in self.pathologies:
            mask = self.csv["class_name"].str.lower().str.contains(pathology.lower())
            if pathology in self.mapping:
                for syn in self.mapping[pathology]:
                    mask |= self.csv["class_name"].str.lower().str.contains(syn.lower())
            labels.append(mask.values)
        self.labels = np.asarray(labels).T
        self.labels = self.labels.astype(np.float32)

        self.csv = self.csv.reset_index()

    def string(self):
        return self.__class__.__name__ + " num_samples={} views={} data_aug={}".format(len(self), self.views, self.data_aug)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return sample

    def get_mask_dict(self, image_name, this_size):
        return path_mask