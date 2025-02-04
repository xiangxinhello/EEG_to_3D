import os
import numpy as np
import torch
from PIL import Image


def load_images(args, idx_val):
	"""Load and preprocess the training, validation and test images.

	Parameters
	----------
	args : Namespace
		Input arguments.
	idx_val : bool
		Indices of the validation images.

	Returns
	-------
	X_train : list of tensor
		Training images.
	X_val : list of tensor
		Validation images.
	X_test : list of tensor
		Test images.

	"""

	import os
	from torchvision import transforms
	from tqdm import tqdm
	from PIL import Image
	from concurrent.futures import ThreadPoolExecutor

	### Define the image preprocesing ###
	preprocess = transforms.Compose([
		transforms.Resize((224,224)),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])

	### Load and preprocess the training and validation images ###
	img_dirs = os.path.join('/root/autodl-tmp/DreamDiffusion/code/eeg_encoding-main', 'image_set', 'training_images')
	image_list = []
	for root, dirs, files in os.walk(img_dirs):
		for file in files:
			if file.endswith(".jpg"):
				image_list.append(os.path.join(root,file))
	image_list.sort()
	X_train = []
	X_val = []


	# for i, image in enumerate(tqdm(image_list)):
	# 	img = Image.open(image).convert('RGB')
	# 	img = preprocess(img)
	# 	if idx_val[i] == True:
	# 		X_val.append(img)
	# 	else:
	# 		X_train.append(img)

	def load_and_preprocess(image):
		img = Image.open(image).convert('RGB')
		img = preprocess(img)
		return img

	with ThreadPoolExecutor() as executor:
		futures = [executor.submit(load_and_preprocess, img) for img in image_list]
		for i,future in tqdm(enumerate(futures), total=len(futures)):
			img = future.result()
			if idx_val[i]:
				X_val.append(img)
			else:
				X_train.append(img)




	### Load and preprocess the test images ###
	img_dirs = os.path.join(args.project_dir, 'image_set', 'test_images')
	image_list = []
	for root, dirs, files in os.walk(img_dirs):
		for file in files:
			if file.endswith(".jpg"):
				image_list.append(os.path.join(root,file))
	image_list.sort()
	X_test = []
	for image in tqdm(image_list):
		img = Image.open(image).convert('RGB')
		img = preprocess(img)
		X_test.append(img)

	### Output ###
	return X_train, X_val, X_test


def load_eeg_data(args, idx_val):
	"""Load the EEG training and test data.

	Parameters
	----------
	args : Namespace
		Input arguments.
	idx_val : bool
		Indices of the validation images.

	Returns
	-------
	y_train : tensor
		Training EEG data.
	y_val : tensor
		Validation EEG data.
	y_test : tensor
		Test EEG data.
	ch_names : list of str
		EEG channel names.
	times : float
		EEG time points.

	"""



	### Load the EEG training data ###
	data_dir = os.path.join('eeg_dataset', 'preprocessed_data', 'sub-'+
		format(args.sub,'1'))
	training_file = 'preprocessed_eeg_training.npy'
	data = np.load(os.path.join('/root/autodl-tmp/DreamDiffusion/code/eeg_encoding_main', data_dir, training_file),
		allow_pickle=True).item()
	y_train = data['preprocessed_eeg_data']
	ch_names = data['ch_names']
	times = data['times']
	# Average across repetitions
	y_train = np.mean(y_train, 1)
	# Extract the validation data
	y_val = y_train[idx_val]
	y_train = np.delete(y_train, idx_val, 0)
	# Convert to float32 and tensor (for DNN training with Pytorch)
	y_train = torch.tensor(np.float32(y_train))
	y_val = torch.tensor(np.float32(y_val))

	### Load the EEG test data ###
	test_file = 'preprocessed_eeg_test.npy'
	data = np.load(os.path.join(args.project_dir, data_dir, test_file),
		allow_pickle=True).item()
	y_test = data['preprocessed_eeg_data']
	# Average across repetitions
	y_test = np.mean(y_test, 1)
	# Convert to float32 and tensor (for DNN training with Pytorch)
	y_test = torch.tensor(np.float32(y_test))

	### Output ###
	return y_train, y_val, y_test, ch_names, times


def create_dataloader(args, time_point, g_cpu, y_train, y_val, y_test):
	"""Put the training, validation and test data into a PyTorch-compatible
	Dataloader format.

	Parameters
	----------
	args : Namespace
		Input arguments.
	time_point : int
		Modeled EEG time point.
	g_cpu : torch.Generator
		Generator object for DataLoader random batching.
	X_train : list of tensor
		Training images.
	X_val : list of tensor
		Validation images.
	X_test : list of tensor
		Test images.
	y_train : float
		Training EEG data.
	y_val : float
		Validation EEG data.
	y_test : float
		Test EEG data.

	Returns
	----------
	train_dl : Dataloader
		Training Dataloader.
	val_dl : Dataloader
		Validation Dataloader.
	test_dl : Dataloader
		Test Dataloader.

	"""

	import torch
	from torch.utils.data import Dataset
	from torch.utils.data import DataLoader
	from scipy.interpolate import interp1d
	import numpy as np

	### Dataset class ###
	class EegDataset(Dataset):
		def __init__(self, y, modeled_time_points, time, transform=None,
			target_transform=None):
			self.modeled_time_points = modeled_time_points
			self.time = time
			# self.X = X
			# if self.modeled_time_points == 'single':
			# 	self.y = y[:,:,self.time]
			# elif self.modeled_time_points == 'all':
			# 	self.y = torch.reshape(y, (y.shape[0],-1))
			self.y = y
			self.transform = transform
			self.target_transform = target_transform

			self.data_len = 512
			self.data_chan = 128

		def __len__(self):
			return len(self.y)

		def __getitem__(self, idx):
			# image = self.X[idx]
			data = self.y[idx]

			if data.shape[-1] > self.data_len:
				idx = np.random.randint(0, int(data.shape[-1] - self.data_len) + 1)

				data = data[:, idx: idx + self.data_len]
			else:
				x = np.linspace(0, 1, data.shape[-1])
				x2 = np.linspace(0, 1, self.data_len)
				f = interp1d(x, data)
				data = f(x2)
			ret = np.zeros((self.data_chan, self.data_len))
			if (self.data_chan > data.shape[-2]):
				for i in range((self.data_chan // data.shape[-2])):
					ret[i * data.shape[-2]: (i + 1) * data.shape[-2], :] = data
				if self.data_chan % data.shape[-2] != 0:
					ret[-(self.data_chan % data.shape[-2]):, :] = data[: (self.data_chan % data.shape[-2]), :]
			elif (self.data_chan < data.shape[-2]):
				idx2 = np.random.randint(0, int(data.shape[-2] - self.data_chan) + 1)
				ret = data[idx2: idx2 + self.data_chan, :]
			# print(ret.shape)
			elif (self.data_chan == data.shape[-2]):
				ret = data
			ret = ret / 10  # reduce an order
			# torch.tensor()
			data = torch.from_numpy(ret).float()


			# if self.transform:
			# 	image = self.transform(image)
			# if self.target_transform:
			# 	target = self.target_transform(data)


			# return image, data
			return data

	### Convert the data to PyTorch's Dataset format ###
	train_ds = EegDataset(y_train, args.modeled_time_points, time_point)

	val_ds = EegDataset(y_val, args.modeled_time_points, time_point)

	test_ds = EegDataset(y_test, args.modeled_time_points, time_point)

	### Convert the Datasets to PyTorch's Dataloader format ###
	train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=g_cpu)
	val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
	test_dl = DataLoader(test_ds, batch_size=test_ds.__len__(), shuffle=False)

	### Output ###
	return train_dl, val_dl, test_dl

from scipy.fftpack import fft, rfft, fftfreq, irfft, ifft, rfftfreq
from PSD_ExtractFeatures import PSD_Etract


def fre_time_split(x):
    eeg = x
    N = x.size(1)
    T = 1.0 / 1000.0
    time = np.linspace(0.0, N * T, N)
    w = rfftfreq(N, T)
    eeg = eeg.numpy()
    eeg_fft = rfft(eeg)
    eeg_fft[:, w < 7] = 0
    eeg_fft[:, w > 71] = 0
    time_eeg = irfft(eeg_fft)
    time_eeg = torch.tensor(time_eeg)
    fre_eeg = PSD_Etract(eeg)
    fre_eeg = torch.tensor(fre_eeg)

    return fre_eeg, time_eeg

def create_dataloader_ldm_b(args, time_point, g_cpu, X_train, X_val, X_test, y_train,
	y_val, y_test):
	"""Put the training, validation and test data into a PyTorch-compatible
	Dataloader format.

	Parameters
	----------
	args : Namespace
		Input arguments.
	time_point : int
		Modeled EEG time point.
	g_cpu : torch.Generator
		Generator object for DataLoader random batching.
	X_train : list of tensor
		Training images.
	X_val : list of tensor
		Validation images.
	X_test : list of tensor
		Test images.
	y_train : float
		Training EEG data.
	y_val : float
		Validation EEG data.
	y_test : float
		Test EEG data.

	Returns
	----------
	train_dl : Dataloader
		Training Dataloader.
	val_dl : Dataloader
		Validation Dataloader.
	test_dl : Dataloader
		Test Dataloader.

	"""

	import torch
	from torch.utils.data import Dataset
	from torch.utils.data import DataLoader
	from scipy.interpolate import interp1d
	import numpy as np
	from transformers import AutoProcessor

	### Dataset class ###
	class EegDataset(Dataset):
		def __init__(self, X, y, modeled_time_points, time, transform=None,
			target_transform=None, image_transform=identity):
			self.modeled_time_points = modeled_time_points
			self.time = time
			self.X = X
			# if self.modeled_time_points == 'single':
			# 	self.y = y[:,:,self.time]
			# elif self.modeled_time_points == 'all':
			# 	self.y = torch.reshape(y, (y.shape[0],-1))
			self.y = y
			self.transform = transform
			self.target_transform = target_transform
			self.processor = AutoProcessor.from_pretrained("/root/autodl-tmp/huggingface/transformers/")
			self.image_transform = image_transform
			self.data_len = 512
			self.data_chan = 128

		def __len__(self):
			return len(self.y)

		def __getitem__(self, idx):
			image = self.X[idx]
			eeg = self.y[idx]

			eeg = eeg[20:460, :]


			eeg_fre_time = eeg
			fre_eeg, time_eeg = fre_time_split(eeg_fre_time)
			fre_eeg, time_eeg = fre_eeg, (time_eeg.t())[20:460, :]

			eeg = np.array(eeg.transpose(0, 1))
			x = np.linspace(0, 1, eeg.shape[-1])
			x2 = np.linspace(0, 1, self.data_len)
			f = interp1d(x, eeg)
			eeg = f(x2)
			eeg = torch.from_numpy(eeg).float()
			image_raw = Image.open(image).convert('RGB')
			image = np.array(image_raw) / 255.0
			image_raw = self.processor(images=image_raw, return_tensors="pt")
			image_raw['pixel_values'] = image_raw['pixel_values'].squeeze(0)

			return {'eeg': eeg, 'image': self.image_transform(image),
                'image_raw': image_raw, 'fre_eeg': fre_eeg, 'time_eeg':time_eeg,

                }

	### Convert the data to PyTorch's Dataset format ###
	train_ds = EegDataset(y_train, args.modeled_time_points, time_point)

	val_ds = EegDataset(y_val, args.modeled_time_points, time_point)

	test_ds = EegDataset(y_test, args.modeled_time_points, time_point)

	### Convert the Datasets to PyTorch's Dataloader format ###
	train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=g_cpu)
	val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
	test_dl = DataLoader(test_ds, batch_size=test_ds.__len__(), shuffle=False)

	### Output ###
	return train_dl, val_dl, test_dl
