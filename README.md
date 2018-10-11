

[![Fish](https://github.com/joeshull/what_the_fish_beta/blob/master/readme_graphics/ex_fish.jpg)](#)

# What The Fish (Beta) 


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


In order to create my equally balanced classes I wrote an <a href="">image scraping script</a> that leverages the <a href="">Google Images Download</a> tool built by hardikvasa.

The Fish Class: Query Google Images for "fish" and download 1000 of the top results. Easy-peasy.
	
	'code here'

The Non-Fish class: Query approx. <a href="https://planspace.org/20170430-sampling_imagenet/">200 "non-fish"</a> categories ranked according to the ImageNet database. 

	'code here'


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

The *Height* and *Width* give us the number of pixels in each dimension. More pixels in each dimension results in a larger picture and higher resolution (pixels per inch) when scale is held constant. 

The *Depth* of 3 gives us three 2D matrices to store the color brightness values for each color (Red, Green, Blue) at each pixel. The values can take
any value from 0 to 255, and their combinations result in over 16-million colors.

Here's an example of an RGB Photo zoomed in to the pixel level

RGB Image Pixels			| RGB Pixels as Integer Values
:-------------------------:|:-------------------------:
<a href="http://archive.xaraxone.com/webxealot/workbook35/rgb-cymk_04.gif"><img src="http://archive.xaraxone.com/webxealot/workbook35/rgb-cymk_04.gif" width="600px" height="400px"></img></a>|<a href="http://archive.xaraxone.com/webxealot/workbook35/rgb-cymk_02.gif"><img src="http://archive.xaraxone.com/webxealot/workbook35/rgb-cymk_02.gif" width="600px" height="400px"></img></a>

#### Grayscaling
I wanted to eliminate color as a variable in order to give the classifier an easier task - identify shape or shading. To do that I used PIL's "L" algorithm to convert the image to grayscale (0-255) and de facto provide us with a 2D matrix.

*Greyscaling Using PIL's 'L' Algorithm*
	
	Gray = R * 299/1000 + G * 587/1000 + B * 114/1000
<div align="Center">
    <img src="https://github.com/joeshull/what_the_fish_beta/blob/master/readme_graphics/gray_smile.png" width="300px" height="300px"></img> 
</div>


Converting to grayscale allows us to do a couple things:
1. Simplify the classification to shape and lighting only
2. Maintain some semblance of sampling density (RGB features at 33px = 3267)


#### Resizing
Once grayscaled, the images were resized to a fixed image size (33x33px). Creating a fixed image size keeps the pixel space consistent at 1089 total pixels.

The image size was picked by visually testing smallest picture size that my human eye could accurately identify a "fish" or "non-fish". 

Fish Image Processing      |  Non-fish Image Processing
:-------------------------:|:-------------------------:
<img src="https://github.com/joeshull/what_the_fish_beta/blob/master/readme_graphics/fish_proc.png" width="600px" height="400px"></img>   |  <img src="https://github.com/joeshull/what_the_fish_beta/blob/master/readme_graphics/nfish_proc.png" width="600px" height="400px"></img>


Logistic regression takes a 2d matrix as input, so each Image was flattened to a vector of length 1089 (33x33).


## Image EDA
With all images converted to a 2D, we can start to explore the classes and their features.

Below is a plot that shows the mean of all pixels for both classes. 

*On the left*, the mean "fish" appears to be of medium light on the borders with a lighter shading in the middle. If you squint, it might even look like a underwater phot of a fish. 

*On the right*, the mean "non-fish" picture appears to have a white border with some object of focus located directly in the center. Google Images seems to favor stock photos (objects on white background) for the first several images in a query. The webscraping script queried 200 words and pulled 5-6 images for each. As you can see, we probably have a large amount of stock photos.

<div align="Left">
	<img src="https://github.com/joeshull/what_the_fish_beta/blob/master/readme_graphics/avgimage1.png" width="1200px" height="600px"></img> 
</div>

Let's take a look at the distribution of values for a random pixel. Pixel 496 is about halfway down the picture and 1 pixel in from the left.

The upper right and left plots are the "average" pictures from before. On the bottom is the Kernel Density Estimation (KDE) for both classes at pixel 496. Here we can see the probability distribution of this pixel and the Expected Value (mean) at that pixel. 

Now in English: At pixel 496, "Fish" images have an expected intensity of ~110 intensity (gray) and we can see the mode takes a value around ~90. "Non-fish" images at this pixel have an expected intensity closer to 190 (light-gray) with the mode >225. 

The probability of an image being labeled "fish" will decrease as intensity increases at Pixel 496. 

<img src="https://github.com/joeshull/what_the_fish_beta/blob/master/readme_graphics/pixel_brightness.png" width="1200px" height="600px"></img>

What does it look like if we subtract the means from each other to find the biggest difference in intensity? E.g. at Pixel 496, the difference between the classes is (~80). 
<img src="https://github.com/joeshull/what_the_fish_beta/blob/master/readme_graphics/histpixdif.png" width="1200px" height="600px"></img>

Once we net the images at each pixel, we can normalize and rescale them back to our 0-255 values for image rendering. 

Below we can see the biggest differences between the two images are at the edges. This makes sense since our fish images are generally grayish across the pixel space, while the non-fish images have a strong white border at the edges.

On the right, I've applied a mask at median (gray-128) to see exactly which pixels will give the classifier the strongest signal.
<img src="https://github.com/joeshull/what_the_fish_beta/blob/master/readme_graphics/netimage.png" width="1200px" height="600px"></img>


Now that we see where the classifier will be getting the strongest signal, let's visualize this across the pixel space. Watch where the dotted "Expected Value" lines get farthest apart *(Hint: It's at the edges as we saw earlier)*

<img src="https://github.com/joeshull/what_the_fish_beta/blob/master/readme_graphics/fishkde.gif" width="1200px" height="600px"></img>


## Logistic Regression

<a href="https://en.wikipedia.org/wiki/Logistic_regression">Logistic Regression</a> is similar to Linear Regression: modeling a dependent variable to the modulation in independent variables. 
While Linear Regression models a continuous output to continuous input
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/8119b3ed1259aa8ff15166488548104b50a0f92e"></img>

The Logistic Regressor models the "Log Odds" (0 and 1) as output to continuous input.

<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/21135f8ddca09553a884ea00e7502d9c3f624385.svg"></img>

The interpretation of the coefficients is similar. In our case, the coefficients should take negative values. 



As we saw in Image EDA, the probability of an image being a "fish" decreases as pixel intensity increases.


For a more in-depth explanation on Logistic Regression, check out this article and the wiki.
https://towardsdatascience.com/logistic-regression-detailed-overview-46c4da4303bc


## Classification Results




ROC & AUC            |  Confusion Matrix
:-------------------------:|:-------------------------:
<img src="https://github.com/joeshull/what_the_fish_beta/blob/master/readme_graphics/rocauc.png" width="600px" height="400px"></img>   |  <img src="https://github.com/joeshull/what_the_fish_beta/blob/master/readme_graphics/conf_matrix.png" width="600px" height="400px"></img>

ROC Curve



Confusion Matrix



Min/Max fish

<img src="https://github.com/joeshull/what_the_fish_beta/blob/master/readme_graphics/MaxMin.png" width="1200px" height="800px"></img>

White_fish Classify

<img src="https://github.com/joeshull/what_the_fish_beta/blob/master/readme_graphics/not_fish.png" width="1200px" height="800px"></img>


## Future Work
	

#### References and Resources
http://archive.xaraxone.com/webxealot/workbook35/rgb-cymk_04.gif
http://archive.xaraxone.com/webxealot/workbook35/rgb-cymk_02.gif

https://github.com/hardikvasa/google-images-download

https://planspace.org/20170430-sampling_imagenet/





