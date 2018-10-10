from pathlib import Path
from sys import argv
import os
import time
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


class ImageToArray(object):
	"""
	Run ImageToArray at top of image directories you want to convert.
	Give it directory to search (must be pictures only) it creates a list of filenames to iterate
	Call .convert_dir
	Run .save_array to save as .npy

	"""
	def __init__(self, directory, resolution = 33):
		self.directory_ = directory
		self.image_list_ = [f for f in os.listdir(directory) if not f.startswith('.')]
		self.resolution_ = resolution
		self.vector_size_ = (self.resolution_**2)
		self.image_array_ = np.empty([len(self.image_list_),self.vector_size_])
		self.image_titles_ = np.empty(len(self.image_list_), dtype=object)

	def convert_dir(self):
		print (f"Converting {len(self.image_list_)} images in {self.directory_}")
		for idx, pic in enumerate(self.image_list_):
			try:
				print (f"Cycle {idx+1}: Opening {pic} and grayscaling")
				image = Image.open(f'{self.directory_}/{pic}').convert('L')
				
				print (f"Resizing to {self.resolution_}x{self.resolution_}")
				image_resized = image.resize((self.resolution_, self.resolution_))
				
				print (f"Flattening to vector length {self.vector_size_}.")
				image_matrix = np.array(image_resized)
				vector = np.ravel(image_matrix)
				
				print ("Adding to .image_array_")
				self.image_array_[idx] = vector
				self.image_titles_[idx] = pic
			
			except OSError:
				print("Error: skipping a non-image file!")
		
		print("Fixing Nans and deleting any all-black pictures.")
		self._fixnans()
		mask = ~(self.image_array_==0).all(1)
		self.image_array_ = self.image_array_[mask]
		self.image_titles_ = self.image_titles_[mask]
		
		print("All done! Run the .save_array() method to save the numpy array to disk or call .image_array_ to use the array.")
		return self
		
	def _fixnans(self):
		self.image_array_ = np.nan_to_num(self.image_array_)

	def save_array(self):
		np.save(self.directory_ , self.image_array_)
		np.save(f"{self.directory_}_titles", self.image_titles_)
		print(f'Saved to {self.directory_}')

def mask_array(mask_image, image_array, mask_size=11):
	mask = Image.open(mask_image).convert('L')
	mask = mask.resize((11,11))
	mask = np.array(mask).ravel()
	mask_insert_len = len(mask)
	for image in image_array:
		image[:mask_insert_len] = mask
	return image_array



if __name__ == '__main__':
	# # #Convert folders of pictures to Numpy Array and Save
	start = time.time()
	
	top_dir = '../images'
	image_dir = ['all_fish','non_fish']
	
	for d in image_dir:
		image_path = f'{top_dir}/{d}'
		ITA = ImageToArray(image_path, 33)
		ITA.convert_dir()
		ITA.save_array()
	
	end = time.time()
	timed = end - start
	print (f"This {len(image_dir)} folder cycle took {timed} seconds")


	# #Mask Fish array with smiley face
	# mask_name = 'mask.png'
	# mask_dir=f'{cwd}/{mask_name}'
	# fishes = np.load('all_fish.npy')

	# masked_fishes = mask_array(mask_dir, fishes)
	# np.save(mask_dir,masked_fishes)










