from google_images_download import google_images_download
from pathlib import Path
import pandas as pd 
import numpy as np 


class GoogleImageDownloader(object):
	"""
	Requires: 
	1. google_images_download @ https://github.com/hardikvasa/google-images-download
	2. Chromedriver
	3. Selenium
		

	GoogleImageDownloader has two methods for downloading images:
		1. From a list of words 2. For a specific keyword.

	Example:
	GoogleImageDownloader().download_images_from_list(wordlist, # pictures, download_diretory)

	GoogleImageDownloader().download_images_keyword(word, # pictures, download_directory)

	"""

	def __init__(self):
		pass

	def download_images_from_list(self, wordlist, limit, image_directory_name):
		"""
		Inputs: wordlist (list of strings)
				limit (integer)
				image_directory_name (directory path as string)

		Outputs: 
				A directory with images downloaded from Google
		"""

		for word in wordlist:
			self.download_images_keyword(word, limit, image_directory_name)


	def download_images_keyword(self, keyword, limit, image_directory_name):
		"""
		Inputs: keyword (string)
				limit (integer)
				image_directory_name (directory path as string)

		Outputs: A directory with images downloaded from Google
		"""

		cwd = str(Path.cwd())
		arguments = {
					"keywords": keyword, "limit" : limit, "output_directory" : f'{cwd}/', 
					"image_directory" : image_directory_name, "print_urls" : True,
					"chromedriver" : "~/.chromedriver/chromedriver"
					}
		response = google_images_download.googleimagesdownload()
		paths = response.download(arguments)
		print(paths)


if __name__ == '__main__':
	words = pd.read_csv('../images/200words.csv', header=None, names=['words'])

	gid = GoogleImageDownloader()
	gid.download_images_from_list(words['words'],6, 'non_fish')
	gid.download_images_keyword('fish', 1000, 'all_fish')
