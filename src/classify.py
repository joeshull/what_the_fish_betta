import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from PIL import Image
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from plot_helper import*


class FishClassify(object):
	"""
	Instantiate this model with data, target, 
	and an SKLearn classifier (default = LogisticRegression).

	Run .fit to get a fitted model.

	Inputs:
		X: a 2-d array
		y: a 1-d array of 0-1 labels with same length as X
		model: (optional) An SKLearn Classification Object.

	Methods:
	fit: returns a fitted model object
	classify photo: returns the classification probability of a photo 
	"""



	def __init__(self, X, y, model=LogisticRegression()):
		self.X = X
		self.y = y
		self.model = model

	def fit(self):
		"""
		Inputs: None
		Outputs: A fitted SKLearn Model
		"""
		self.model = self.model.fit(self.X, self.y)
		return self.model


	def classify_photo(self,file_path):
		"""
		Inputs: A string path to image file.
		Outputs: The binary classification probability
		"""
		photo_fish = np.array(Image.open(file_path).convert('L').resize((33,33))).ravel().reshape(1,-1)
		prob = self.model.predict_proba(photo_fish)
		print(prob)


if __name__ == '__main__':

	#Load Fish pictures array
	X_fish = np.load('../data/all_fish.npy')
	X_nfish = np.load('../data/non_fish.npy')
	files_fish = np.load('../data/all_fish_titles.npy')


	#Load Non-fish pictures
	y_fish = np.ones(len(X_fish))
	y_nfish = np.zeros(len(X_nfish))
	files_nfish = np.load('../data/non_fish_titles.npy')

	#Make X and y
	X = np.append(X_fish, X_nfish, axis=0)
	y = np.append(y_fish, y_nfish)
	y_files = np.append(files_fish, files_nfish)


	#Make Holdout Set
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

	#Fit
	fishmodel = FishClassify(X_train, y_train)
	fishmodel = fishmodel.fit()

	# # # Classify a stock photo of a fish.
	# # #----------------------------------
	# classify_photo('../images/fish_white.jpg', model)


	##Plot KDEs for a chosen pixel on each class
	###---------------------------------
	# for i in range(1089):
	# 	fig = plt.figure(figsize=(16,8))
	# 	ax = fig.add_subplot(212)
	# 	ax1 = fig.add_subplot(221)
	# 	ax2 = fig.add_subplot(222)
	# 	plot_kdes(X_fish, X_nfish, i, ax, ax1, ax2)
	# 	plt.savefig(f'savefig/{i:04}')
	# 	plt.close()

	## Plot Histogram of differences.
	avg_fish = X_fish.mean(axis=0)
	avg_nfish = X_nfish.mean(axis=0)
	net_image = np.absolute(avg_fish - avg_nfish)
	fig = plt.figure(figsize=(4,4))
	fig.suptitle("Histogram of Pixel Differences", fontsize=24, color="Blue")
	ax = fig.add_subplot(111)
	ax.hist(net_image)
	ax.set_xlabel("Intensity Difference (Absolute)", fontsize=16)
	ax.set_ylabel("Pixel Count (Total 1089)", fontsize=16)
	plt.show()






	# Plot net picture
	avg_fish = X_fish.mean(axis=0)
	# avg_nfish = X_nfish.mean(axis=0)
	# net_image = np.absolute(avg_fish - avg_nfish)
	# net_image = net_image - net_image.min()
	# net_image = net_image/net_image.max()
	# net_image = net_image*255


	# fig = plt.figure(figsize=(12,6))
	# fig.suptitle("Image Class Difference. (White = Largest Difference)", fontsize=24, color="Blue")
	# ax1 = fig.add_subplot(121)
	# plot_vector_image(net_image, ax1, "Net Image")
	# net_image_masked = np.array([255 if x>128 else 0 for x in net_image])
	# ax2 = fig.add_subplot(122)
	# plot_vector_image(net_image_masked, ax2, "Binary Mask at Median")     

	# plt.show()




	# # # #Plot average fish, average non-fish
	# # ## -----------------------------------
	# fig = plt.figure(figsize=(8,6))
	# ax1 = fig.add_subplot(121)
	# ax2 = fig.add_subplot(122)
	# plot_avg_photo(X_fish, X_nfish, ax1, ax2)
	# plt.show()




	# ##Plot fishiest fish, non-fishiest non-fish
	# ##--------------------------------------
	# plot_top_photos(X, y_files, model)



	# #PLOT ROC CURVE
	# #------------------------
	# plot_roc(model, X_test, y_test)


	# #PLOT CONFUSION MATRIX
	# ------------------------
	# cnf_matrix = confusion_matrix(y_test, model.predict(X_test))
	# fig = plt.figure(figsize=(12,8))
	# ax = fig.add_subplot(111)
	# ax.grid(False)
	# class_names = ['fish', 'not fish']
	# plot_confusion_matrix(cnf_matrix, ax, classes=class_names,normalize=True,
 #                      title='Confusion matrix, with normalization')
	# plt.show()

