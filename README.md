# What The Fish (Beta) 

An exploration in image classification using Logistic Regression

Tools: [Python , Numpy, Python Image Library, SKLearn]


## Why The Fish?
My friend likes to fish the rivers and lakes in Colorado, but often doesn't know what fish he's brought in. He asked for an ML tool that will identify fish species, but this is Capstone 1 so we're starting at the ground floor.


## Goals
Attempt a "fish or not-a-fish" image classification with Logistic Regression trained on ~1500 33x33px grayscaled images.


## Image Acquisition

Downloaded ~1000 pictures for "fish"  and 1000 pictures of "non-fish" queried from 200 ImageNet categories 

Fish 						| Fish Eater (Non-fish) 
:-------------------------:|:-------------------------:
<img src="https://github.com/joeshull/what_the_fish_beta/blob/master/readme_graphics/ex_fish.jpg" width="600px" height="300px"></img>	|	<img src="https://github.com/joeshull/what_the_fish_beta/blob/master/readme_graphics/ex_bear.jpg" width="600px" height="400px"></img>


## Image Processing

The images I downloaded started as RGB images in either JPEG or PNG form.

Here's an example of an RGB Photo zoomed in to the pixel level

<div align="Left">
    <img src="http://archive.xaraxone.com/webxealot/workbook35/rgb-cymk_04.gif" width="600px" height="600px"></img> 
</div>



courtesy of xaraxone.com

<div align="Left">
    <img src="http://archive.xaraxone.com/webxealot/workbook35/rgb-cymk_02.gif" width="600px" height="600px"></img> 
</div>
courtesy of xaraxone.com


I wanted to eliminate color as a variable to see if the classifier could simply identify some shape in the image. To do that I used PIL's "L" algoright to convert the image to grayscale and de facto provide us with a 2D matrix.

Greyscaled Using PIL's 'L' Algorithm
	L = R * 299/1000 + G * 587/1000 + B * 114/1000

<div align="Left">
    <img src="https://github.com/joeshull/what_the_fish_beta/blob/master/readme_graphics/gray_smile.png" width="300px" height="300px"></img> 
</div>

Fish Image Processing      |  Non-fish Image Processing
:-------------------------:|:-------------------------:
<img src="https://github.com/joeshull/what_the_fish_beta/blob/master/readme_graphics/fish_proc.png" width="600px" height="400px"></img>   |  <img src="https://github.com/joeshull/what_the_fish_beta/blob/master/readme_graphics/nfish_proc.png" width="600px" height="400px"></img>


Logistic regression takes a 2d matrix as input, so each picture was flattened to a vector of length 1089 (33x33).


## Image EDA
<div align="Left">
	<img src="https://github.com/joeshull/what_the_fish_beta/blob/master/readme_graphics/avgimage1.png" width="1200px" height="600px"></img> 
</div>


Talk about Google Image Query Bias



-Graphic of Brightness distribution at a random pixel

<img src="https://github.com/joeshull/what_the_fish_beta/blob/master/readme_graphics/pixel_brightness.png" width="1200px" height="600px"></img>
This probability distribution at each pixel will inform the classifier

What does it look like if we subtract the means from each other to find the biggest difference? E.g. If at Pixel 515, the means of both classes are similar at (128) the difference would be 0. Similarly, if the one class is, on average, a dark gray (75) but the other is brighter, we would see a large difference. 
<img src="https://github.com/joeshull/what_the_fish_beta/blob/master/readme_graphics/histpixdif.png" width="1200px" height="600px"></img>

Once we net the images at each pixel, we can standardize and rescale them back to our 0-255 values for image rendering. Below we can see the biggest differences between the two images are at the edges. This makes sense since our fish images are generally grayish across the pixel space, while the non-fish images have a strong white border at the edges.

On the right, I've applied a mask at median (gray-128) to see exactly which pixels will give the classifier the strongest signal. The white pixels have a difference 
<img src="https://github.com/joeshull/what_the_fish_beta/blob/master/readme_graphics/netimage.png" width="1200px" height="600px"></img>


<img src="https://github.com/joeshull/what_the_fish_beta/blob/master/readme_graphics/fishkde.gif" width="1200px" height="600px"></img>

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


## Conclusion



