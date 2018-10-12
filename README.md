

[![Fish](https://github.com/joeshull/what_the_fish_beta/blob/master/readme_graphics/bettafish.jpg)](#)

# What The Fish (Betta) 


An exploration in image classification using Logistic Regression

	Tools: [Python, Pandas, Numpy, Python Image Library, SKLearn]


## Why The Fish?
My friend likes to fish the rivers and lakes in Colorado, but often doesn't know what fish he's brought in. He asked for an ML tool that will identify fish species, but this is Capstone 1 so we're starting at the ground floor.


## How the Fish?

**[Image Acquisition](#image-acquisition):** Create a balanced set of "Fish" and "Non-Fish" images.

**[Image Processing](#image-processing):** Convert images to a format compatible with Logistic Regression.

**[Image EDA](#image-eda):** Explore the features and what the classifier will see. 

**[Logistic Regression](#logistic-regression):** Fit SKLearn's Logistic Regressor with 1500 Fish/Non-Fish labeled images.

**[Classification Results](#classification-results):** Report the results on a holdout set of 500 images.


Now that we have a high-level view of the plan, let's dive in! (don't worry, these fish don't bite).


## Image Acquisition


In order to create my image classes I wrote an <a href="">image scraping script</a> that leverages the <a href="">Google Images Download</a> tool built by @hardikvasa.

The Fish Class: Query Google Images for "Fish" and download ~1000 of the top results. Easy-peasy.
	
	gid = GoogleImageDownloader()	
	gid.download_images_keyword('fish', 1000, 'all_fish')

The Non-Fish class: Query approx. <a href="https://planspace.org/20170430-sampling_imagenet/">200 "non-fish"</a> categories ranked according to the ImageNet database. 

	gid.download_images_from_list(words['words'],5, 'non_fish')


Once downloaded, I manually screened the folders to make sure the classes were accurately labeled and there were no fish images in the "non-fish" set and vice versa.

Fish 						| Non-Fish (fish eater)
:-------------------------:|:-------------------------:
<img src="https://github.com/joeshull/what_the_fish_beta/blob/master/readme_graphics/ex_fish.jpg" width="600px" height="300px"></img>	|	<img src="https://github.com/joeshull/what_the_fish_beta/blob/master/readme_graphics/ex_bear.jpg" width="600px" height="400px"></img>

## Image Processing
In general, the images arrived as an RGB image in JPG or PNG format. 

Numerically speaking, RGB images are 3D matrices with shape: 
**Width: columns|
Height: rows|
Depth:  3**

The *Height* and *Width* give us the number of pixels in each dimension. More pixels in each each dimension give us a larger picure or higher resolution (pixels per inch).

The *Depth* of 3 gives us three 2D matrices to store the color brightness values for each color (Red, Green, Blue) at each pixel. The values can take
any value from 0 to 255, and their combinations result in over 16-million colors.

Here's an example of an RGB Photo zoomed in to the pixel level

RGB Image Pixels			| RGB Pixels as Integer Values
:-------------------------:|:-------------------------:
<a href="http://archive.xaraxone.com/webxealot/workbook35/rgb-cymk_04.gif"><img src="http://archive.xaraxone.com/webxealot/workbook35/rgb-cymk_04.gif" width="600px" height="400px"></img></a>|<a href="http://archive.xaraxone.com/webxealot/workbook35/rgb-cymk_02.gif"><img src="http://archive.xaraxone.com/webxealot/workbook35/rgb-cymk_02.gif" width="600px" height="400px"></img></a>

Now that we know a little about RGB values, it's time to process the pictures! (Script<a href="https://github.com/joeshull/what_the_fish_beta/blob/master/src/image_proc.py"> **here**</a>)

#### Grayscaling
I wanted to eliminate color as a variable in order to give the classifier an easier task - identify shape or shading. To do that I used PIL's "L" algorithm to convert each image to grayscale (0-255) and de facto provide us with a 2D matrix.

*Greyscaling Using PIL's 'L' Algorithm*
	
	Gray = R * 299/1000 + G * 587/1000 + B * 114/1000

Grayscale makes me smile.
<div align="Center">
    <img src="https://github.com/joeshull/what_the_fish_beta/blob/master/readme_graphics/gray_smile.png" width="300px" height="300px"></img> 
</div>


Converting to grayscale allows us to do a couple things:
1. Simplify the classification to shape and lighting only
2. Maintain some semblance of sampling density (RGB features at 33px = 3267)


#### Resizing
Once grayscaled, the images were resized to a fixed image size (33x33px). Creating a fixed image size keeps the pixel space consistent at 1089 total pixels.

I picked the image size by visually testing smallest picture size at which my human eye could accurately identify a "fish" or "non-fish". 


Fish Image Processing      |  Non-fish Image Processing
:-------------------------:|:-------------------------:
<img src="https://github.com/joeshull/what_the_fish_beta/blob/master/readme_graphics/fish_proc.png" width="600px" height="400px"></img>   |  <img src="https://github.com/joeshull/what_the_fish_beta/blob/master/readme_graphics/nfish_proc.png" width="600px" height="400px"></img>


Logistic regression takes a 2d matrix as input, so each Image was flattened to a vector of length 1089 (33x33).


## Image EDA
With all images converted to a 2D array, we can start to explore each class and their features.

Below is a plot that shows the mean of all pixels for both classes. 

*On the left*, the mean "fish" appears to be of lower intensity on the borders with a brighter shading in the middle. If you squint, it might even look like a underwater photo of a fish. 

*On the right*, the mean "non-fish" picture appears to have a white border with some object of focus located directly in the center. Google Images seems to favor stock photos (objects on white background) for the first several images in a query. The webscraping script queried 200 words and pulled 5-6 images for each. As you can see, we probably have a large amount of stock photos.

<div align="Left">
	<img src="https://github.com/joeshull/what_the_fish_beta/blob/master/readme_graphics/avgimage1.png" width="1200px" height="600px"></img> 
</div>

Let's take a look at the distribution of values for a random pixel. Pixel 496 is about halfway down the picture and 1 pixel in from the left.

The upper right and left plots are the "average" pictures from before. On the bottom is the Kernel Density Estimation (KDE) for both classes at pixel 496.  Here we can see the probability distribution of this pixel and the Expected Value (mean) at that pixel. 

Now in English: At pixel 496, "Fish" images have an expected intensity of ~110 (gray) and we can see the mode takes a value of ~90. "Non-fish" images at this pixel have an expected intensity closer to 190 (light-gray) with the mode at ~250. 

The probability of an image being labeled "fish" will decrease as intensity increases above the combined mean at Pixel 496. 

<img src="https://github.com/joeshull/what_the_fish_beta/blob/master/readme_graphics/pixel_brightness.png" width="1200px" height="600px"></img>

What does it look like if we subtract the means from each other to find the biggest difference in intensity? E.g. at Pixel 496, the difference between the classes is (~80). 
<img src="https://github.com/joeshull/what_the_fish_beta/blob/master/readme_graphics/histpixdif.png" width="1200px" height="600px"></img>

Once we net the images at each pixel, we can normalize and rescale them back to our 0-255 values for image rendering. 

Below we can see the biggest differences between the two images are at the edges. 
On the right, I've applied a binary mask at the median (gray-128) to see exactly which pixels will give the classifier the strongest signal.
<img src="https://github.com/joeshull/what_the_fish_beta/blob/master/readme_graphics/netimage.png" width="1200px" height="600px"></img>


Now that we see where the classifier will be getting the strongest signal, let's visualize this across the pixel space. Watch where the dotted "Expected Value" lines get farthest apart.

<img src="https://github.com/joeshull/what_the_fish_beta/blob/master/readme_graphics/fishkde.gif" width="1200px" height="600px"></img>


## Logistic Regression
To classify the pictures, I'm using classic <a href="https://en.wikipedia.org/wiki/Logistic_regression">Logistic Regression</a>. Logistic Regression is similar to Linear Regression: modeling a dependent variable response to the change in independent variables. 

Here's the big difference:
While Linear Regression models a continuous output to continuous input

<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/8119b3ed1259aa8ff15166488548104b50a0f92e"></img>

The Logistic Regressor models the "Log Odds" (0.0-1.0) as output to continuous input.

<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/21135f8ddca09553a884ea00e7502d9c3f624385"></img>

This gives us a probability classifier for two classes: "Fish" or "Non-Fish" in our case.

Since our feature space is so large (1089) relative to the sample size, I used an <a href="https://en.wikipedia.org/wiki/Lasso_(statistics)">L1 regularization</a> to penalize the model on the absolute value of the coefficients. This incentivizes the model to use the strongest features and eliminate the features which are not contributing. I also scaled and standardized the data (For all columns, subtract the mean and divide by the standard deviation) using SKLearn's Standard Scaler. This will make our coefficients more stable and interpretable. 

The interpretation of the coefficients is similar to that of linear regression. In our case, when the coefficients of a given pixel is negative, the probability of that image being a fish decreases as pixel intensity increases. The opposite is also true. 

Let's look at where our coefficients are negative and positive. (White = +, Black = -)

<img src="https://github.com/joeshull/what_the_fish_beta/blob/master/readme_graphics/coef.png"></img>

When we look at our <a href="https://github.com/joeshull/what_the_fish_beta/blob/master/readme_graphics/avgimage1.png">Average Picture</a> for each class we see that "Non-Fish" are generally brighter at the edges, while "Fish" pictures are brighter in the center. The direction of our coefficients speak to this relationship: As pixel intensity increases around the edges, we generally see a negative value for it's relative "fishiness". Conversely, in the middle, we see some positive correlation with pixel intensity and "fishiness".

Of note: Though the "Non-Fish" images were generally brighter at the edges, the positive coefficients at the top edge correspond to the few areas where the "Fish" images had a higher pixel intensity than the "non-fish" images.


*For a more in-depth explanation on Logistic Regression, check out this <a href="https://towardsdatascience.com/logistic-regression-detailed-overview-46c4da4303bc">article</a> and the <a href="https://en.wikipedia.org/wiki/Logistic_regression">wiki</a>.*



## Classification Results
Now that we've explored the data and our model, let's look at the results!


ROC & AUC            |  Confusion Matrix
:-------------------------:|:-------------------------:
<img src="https://github.com/joeshull/what_the_fish_beta/blob/master/readme_graphics/rocauc.png" width="600px" height="400px"></img>   |  <img src="https://github.com/joeshull/what_the_fish_beta/blob/master/readme_graphics/conf_matrix.png" width="600px" height="400px"></img>

Not bad! On the left, we see that the classifier got an Area Under the Curve of almost 78%. It was able to catch 71% of  "Fish" pictures and 77% on "Non-Fish" pictures from a holdout set of 500 pictures that it hadn't previously seen.

Let's see the most-probable samples from each class. 

<img src="https://github.com/joeshull/what_the_fish_beta/blob/master/readme_graphics/MaxMin.png" width="1200px" height="800px"></img>

As expected, it picked up on the difference in brightness between the stock photos and the darker sea-scape pictures. The "fishiest fish" is an image with a bright center and dark border, and the opposite is true for the "non-fish".


And just to make sure we haven't accidentally discovered the secret fish-detection powers of Logistic Regression.....

<img src="https://github.com/joeshull/what_the_fish_beta/blob/master/readme_graphics/not_fish.png" width="1200px" height="800px"></img>


## Future Work
Going forward, I would like to build an image-classification neural net and app so that my friends can classify fish species.

Here are a few things I'll need to do:
- [x] Get a list of fish species in Colorado
- [x] Download at least 500 photos for each species of fish to classify
- [x] Retrain a MobileNet classifier to classify fish species
- [ ] Train it better, or make a new model so it classifies accurately
- [ ] Make an app somehow
	
## AUTHOR
[auth]: #author 
You can follow me on [twitter](https://twitter.com/joeyshull) or just [email](mailto:joseph.shull@gmail.com) me.

## ACKNOWLEDGMENTS
[acc]: acknowledgments

List of people that I would like to thank:

- Jamie Sloat for her endless support.
- Rob Troup for his endless stream of great data science ideas.
- Frank Burkholder for great graphic ideas.
- Michael Dyer for tech geekery and tech support.


Copyright © 2018 Joe Shull



