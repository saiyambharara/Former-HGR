from __future__ import print_function, division
import os
import argparse
import enum
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd




DatasetsMeanAndStd = {
	
	"nv":{
		25: {'mean': [0.5000, 0.5000,0.5000], 'std': [0.2013, 0.2013, 0.2013]}
	}

}



class ClassesNV(enum.Enum):
	class_01 = 0
	class_02 = 1
	class_03 = 2
	class_04 = 3
	class_05 = 4
	class_06 = 5
	class_07 = 6
	class_08 = 7
	class_09 = 8
	class_10 = 9
	class_11 = 10
	class_12 = 11
	class_13 = 12
	class_14 = 13
	class_15 = 14
	class_16 = 15
	class_17 = 16
	class_18 = 17
	class_19 = 18
	class_20 = 19
	class_21 = 20
	class_22 = 21
	class_23 = 22
	class_24 = 23
	class_25 = 24




classes_map = {
	
	"nv": {25:ClassesNV}
}

def default_loader(path, transform):
	return transform(Image.open(path))


# def video_frames_loader(path, transform):
# 	images = []
# 	for image in sorted(os.listdir(path)):
# 		images.append(transform((Image.open(os.path.join(path,image))).convert('RGB')))
# 	return torch.stack(images,dim=1)
		
def video_frames_loader(path, transform):
	mean_im = []
	images = 0.0
	count = 0
	for image in sorted(os.listdir(path)):
		# images.append(transform(Image.open(os.path.join(path,image))))
		i=0
		if (count)%5==0:
			images =0.0
			img = Image.open(os.path.join(path,image))
			if len(img.size) ==2:
				img = img.convert('RGB')
				# img = torch.from_numpy(img)
				# img = torch.stack([img,img,img],0)
				# img = torch.transpose(img,0,2)
				# img = img.numpy()
			w,h= img.size
			# if h==256:
			# 	img = img.crop((50,18,250,250))
			# else:
			# 	img = img.crop((100,36,500,500))
		images= images+(transform(img))
		if count%5==4:
			mean_im.append(images/5)
		# 
		count = count+1
	return torch.stack(mean_im,dim=1)

class ClassificationDataset(Dataset):

	def __init__(self, split, train):
		super().__init__()
		# print("Dataset:",DatasetsLocation.get(args.dataset))
		dataset="nv"
		no_of_classes=25
		DatasetsLocation = {'nv': {25: split}} 
		self.base_dir = DatasetsLocation.get(dataset).get(no_of_classes)
		
		# print(dataset.upper())
		self.image_dir = os.path.join("data", dataset.upper(), self.base_dir)
		
		self.image_paths = []
		self.labels = []
		self.sub_names = []
		self.loader = video_frames_loader
		
		
		subjects_or_folds = os.listdir(self.image_dir)
		
			
		for i in range(0, len(subjects_or_folds)):
			classes = subjects_or_folds[i]
			label = 0
			no_of_videos = 0
			for videos in os.listdir(os.path.join(self.image_dir, classes)):
				no_of_videos = no_of_videos + 1
				address = os.path.join(self.image_dir,classes,videos)
				self.image_paths.append(address)
				
				if dataset == "nv":
					self.labels.append(classes_map.get(dataset).get(no_of_classes)[classes].value)
				else:
					self.labels.append(label)
				label += 1
					# print("subject",subject,"no_of_videos",no_of_videos)

#         if abs(((len(self.image_paths) // args.batch_size) * args.batch_size) - len(self.image_paths)) == 1 and len(self.image_paths) != 1:
#             del self.image_paths[-1]
#             del self.labels[-1]
		if len(subjects_or_folds)==1:
			self.test_no_of_videos = no_of_videos
			self.test_subject_name = subjects_or_folds[0]
		self.transform = transforms.Compose([])
		print("train:",train)
		if dataset == 'nv' and train: 
			self.transform.transforms.append(transforms.Resize((112, 112)))
			# self.transform.transforms.append(transforms.RandomResizedCrop(112))
			self.transform.transforms.append(transforms.RandomVerticalFlip())
			# self.transform.transforms.append(transforms.ColorJitter())
			# self.transform.transforms.append(transforms.RandomRotation(degrees=15))
			# self.transform.transforms.append(transforms.RandomRotation(degrees=30))
			# self.transform.transforms.append(transforms.RandomRotation(degrees=45))
			# self.transform.transforms.append(transforms.Grayscale())
			# self.transform.transforms.append()
		else:
			self.transform.transforms.append(transforms.Resize((112, 112)))
			# self.transform.transforms.append(transforms.Grayscale())
			# self.transform.transforms.append(transforms.Lambda(lambda x:x.repeat(3,1,1) if x.size(0)==1 else x))
		
		self.transform.transforms.append(transforms.ToTensor())
		self.transform.transforms.append(
			transforms.Normalize(DatasetsMeanAndStd.get(dataset).get(no_of_classes).get('mean'),
								  DatasetsMeanAndStd.get(dataset).get(no_of_classes).get('std')))


		print(f'Number Of Videos in {dataset.upper()} dataset : {len(self.image_paths)}')

	def __getitem__(self, index):
		path = self.image_paths[index]

		# print("sub_name:",sub_name)
		label = self.labels[index]
		img = self.loader(path, self.transform)
		# print("img:",img.size())
		return {'image': img, 'label': label}

	def __len__(self):
		return len(self.image_paths)


def train_data_loader():
    # image_size = 112
    # train_transforms = torchvision.transforms.Compose([GroupRandomSizedCrop(image_size),
    #                                                    GroupRandomHorizontalFlip(),
    #                                                    Stack(),
    #                                                    ToTorchFormatTensor()])
    
        
    train_data = ClassificationDataset(train=True, split="train")

    # print(train_data.transform)
    # exit()

        
    return train_data


def test_data_loader():
    
        
    test_data = ClassificationDataset(train=False, split="test")
    return test_data
