import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from PIL import Image
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from plot_helper import*





def classify_photo(file_path, model):
	photo_fish = np.array(Image.open(file_path).convert('L').resize((33,33))).ravel().reshape(1,-1)
	prob = model.predict_proba(photo_fish)
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
	model = LogisticRegression()
	model = model.fit(X_train, y_train)

	# # # Classify a stock photo of a fish.
	# # #----------------------------------
	# classify_photo('../images/fish_white.jpg', model)


	# ##Plot KDEs for a chosen pixel on each class
	# ###---------------------------------
	# for i in range(1089):
	# 	fig = plt.figure(figsize=(16,8))
	# 	ax = fig.add_subplot(212)
	# 	ax1 = fig.add_subplot(221)
	# 	ax2 = fig.add_subplot(222)
	# 	plot_kdes(X_fish, X_nfish, i, ax, ax1, ax2)
	# 	plt.savefig(f'savefig/{i:04}')
	# 	plt.close()



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

