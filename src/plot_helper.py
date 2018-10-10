import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import itertools
from PIL import Image
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib as mpl
from transparent_imshow import transp_imshow
font_size = 24
mpl.rcParams.update({'font.size': font_size})
mpl.rcParams['xtick.labelsize'] = font_size-5
mpl.rcParams['ytick.labelsize'] = font_size-5
plt.style.use('fivethirtyeight')



def plot_roc(fitted_model, X, y):
	probs = fitted_model.predict_proba(X)
	fpr, tpr, thresholds = roc_curve(y, probs[:,1])
	auc_score = round(roc_auc_score(y,probs[:,1]), 4)
	plt.plot(fpr, tpr, label= f'AUC = {auc_score}')
	plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k',
         label='Luck')
	plt.xlabel("False Positive Rate (1-Specificity)")
	plt.ylabel("True Positive Rate (Sensitivity, Recall)")
	plt.title("ROC plot of 'Fish, Not A Fish' with Logistic Regression")
	plt.legend()
	plt.show()

def plot_confusion_matrix(cm, ax, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """
    p = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title,fontsize=24)
    
    plt.colorbar(p)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=0)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)
   
    if normalize:
        cm = np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis],2)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 1.5
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, cm[i, j],
                 horizontalalignment="center", size = 24,
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    ax.set_ylabel('True label',fontsize=24)
    ax.set_xlabel('Predicted label',fontsize=24)

def plot_avg_photo(X1, X2, ax1, ax2):
	avg_fish = X1.mean(axis=0)
	avg_nfish = X2.mean(axis=0)
	plot_vector_image(avg_fish, ax1, "Average Fish")
	plot_vector_image(avg_nfish, ax2, "Average Non-Fish")
	return ax1, ax2

def plot_top_photos(X_all, y_files, model):
	fig = plt.figure(figsize=(13,8))
	ax1 = fig.add_subplot(221)
	ax2 = fig.add_subplot(222)
	ax1_1 = fig.add_subplot(223)
	ax2_1 = fig.add_subplot(224)
	probs = model.predict_proba(X_all)
	top_fish = np.argmax(probs[:,1])
	top_fishog = Image.open(f'../images/all_fish/{y_files[top_fish]}')
	top_nfish = np.argmin(probs[:,1])
	top_nfishog = Image.open(f'../images/non_fish/{y_files[top_nfish]}')
	print(f'top_fish = {y_files[top_fish]}')
	print(f'top_nfish = {y_files[top_nfish]}')
	plot_vector_image(X_all[top_fish],ax1, "Fishiest Fish")
	plot_vector_image(X_all[top_nfish],ax2, "Least Fishy Non-Fish")
	ax1_1.imshow(np.array(top_fishog))
	ax2_1.imshow(np.array(top_nfishog))
	plt.tight_layout()
	plt.show()


def plot_vector_image(vector, ax, title, square_size = 33):
	image = vector.reshape(33,33)
	ax.imshow(image, cmap='gray')
	ax.set_title(title)
	ax.set_axis_off()
	return ax



def plot_kdes(X1, X2, pixel, ax1, ax_f, ax_nf):
	avg_X1 = X1.mean(axis=0)
	avg_X2 = X2.mean(axis=0)
	

	mask = np.zeros(len(avg_X1))
	mask[pixel] = 1
	mask = mask.reshape(33,33)

	ax_f, ax_nf = plot_avg_photo(X1, X2, ax_f, ax_nf)
	transp_imshow(mask,ax_f, cmap="Reds")
	transp_imshow(mask,ax_nf,cmap="Reds")


	x1k = stats.gaussian_kde(X1[:,pixel])
	x2k = stats.gaussian_kde(X2[:,pixel])
	x_range = np.arange(0,256)
	y1 = x1k.pdf(x_range)
	y2 = x2k.pdf(x_range)
	ax1.plot(x_range, y1, label='Fish Picture')
	ax1.plot(x_range, y2, label='Non-Fish Picture')
	ax1.set_xticks([0,128,255])
	ax1.set_xticklabels(['Black','Gray' ,'White'])
	ax1.set_ylim(bottom=0.00, top=0.0125)
	ax1.set_ylabel('Probability')
	ax1.set_title(f"KDE at Pixel {pixel} ")
	ax1.legend(loc=2)

def skimage_resize_demo():
	mask = Image.open('mask.png')
	mask = mask.convert('L')
	mask = np.array(mask)
	mask[mask<128] = 0
	mask[mask>=128] = 255
	# mask = resize(mask, (11,11), anti_aliasing= True)



	image = Image.open('bear.jpg')
	image_c = image.convert('RGB')
	image_c = np.array(image_c)
	image = image.convert('L')
	image_resized = image.resize((33,33))
	image_resized = np.array(image_resized)
	image = np.array(image)



	image_rescaled = rescale(image, 1.0 / 36.0, anti_aliasing=True)
	
	# image_resized = image.resize((33,33))
	# image_resized[:11,:11] = mask


	fig, axes1 = plt.subplots(nrows=2, ncols=1)
	ax2 = fig.add_subplot(223)
	ax3 = fig.add_subplot(224)

	ax1 = axes1.ravel()


	ax1[0].imshow(image_c, cmap='gray')
	ax1[0].set_title("Original image")
	
	ax1[1].axis('off')


	ax2.imshow(image_rescaled, cmap='gray')
	ax2.set_title("Grayscaled / Rescaled image")

	ax3.imshow(image_resized, cmap='gray')
	ax3.set_title("PIL Grayscaled / Resized image")


	ax1[0].set_xlim(0, 7360)
	ax1[0].set_ylim(4912, 0)
	plt.tight_layout()
	plt.show()


	





if __name__ == '__main__': 
	pass
