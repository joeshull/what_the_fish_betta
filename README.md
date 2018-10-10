# What The Fish (Beta) 

An exploration in image classification using Logistic Regression

Tools: [Python , Numpy, Python Image Library, SKLearn]


## Why The Fish?
My friend likes to fish the rivers and lakes in Colorado, but often doesn't know what fish he's brought in. He asked for an ML tool that will identify fish species, but this is Capstone 1 so we're starting at the ground floor.


## Goals
Attempt a "fish or not-a-fish" image classification with Logistic Regression trained on ~1500 33x33px grayscaled images.


## Image Acquisition

Downloaded ~1000 pictures for "fish"  and 1000 pictures of "non-fish" queried from 200 ImageNet categories 

Fish 						| Fish Eater (Non-fish). 
:-------------------------:|:-------------------------:
<img src="https://github.com/joeshull/what_the_fish_beta/blob/master/readme_graphics/ex_fish.jpg" width="400px" height=200px"></img>	|	<img src="https://github.com/joeshull/what_the_fish_beta/blob/master/readme_graphics/ex_bear.jpg" width="400px" height="350px"></img>


## Image Processing

The images I downloaded started as RGB images in either JPEG or PNG form.

Here's an example of an RGB Photo zoomed in to the pixel level

<div align="Left">
    <img src="http://archive.xaraxone.com/webxealot/workbook35/rgb-cymk_04.gif" width="400px" height="400px"></img> 
</div>



courtesy of xaraxone.com

<div align="Left">
    <img src="http://archive.xaraxone.com/webxealot/workbook35/rgb-cymk_01.gif" width="400px" height="400px"></img> 
</div>
courtesy of xaraxone.com


I wanted to eliminate color as a variable to see if the classifier could simply identify some shape in the image. To do that I used PIL's "L" algoright to convert the image to grayscale and de facto provide us with a 2D matrix.

Greyscaled Using PIL's 'L' Algorithm
	L = R * 299/1000 + G * 587/1000 + B * 114/1000

<div align="Left">
    <img src="https://github.com/joeshull/what_the_fish_beta/blob/master/readme_graphics/gray_smile.png"></img> 
</div>

Fish Image Processing      |  Non-fish Image Processing
:-------------------------:|:-------------------------:
<img src="https://github.com/joeshull/what_the_fish_beta/blob/master/readme_graphics/fish_proc.png" width="400px" height="400px"></img>   |  <img src="https://github.com/joeshull/what_the_fish_beta/blob/master/readme_graphics/nfish_proc.png" width="400px" height="400px"></img>


Logistic regression takes a 2d matrix as input, so each picture was flattened to a vector of length 1089 (33x33).


## Image EDA
<div align="Left">
	<img src="https://github.com/joeshull/what_the_fish_beta/blob/master/readme_graphics/avgimage1.png" width="800px" height="400px"></img> 
</div>


Talk about Google Image Query Bias



-Graphic of Brightness distribution at a random pixel

<img src="https://github.com/joeshull/what_the_fish_beta/blob/master/readme_graphics/pixel_brightness.png" width="800px" height="600px"></img>
This probability distribution at each pixel will inform the classifier


<img src="https://github.com/joeshull/what_the_fish_beta/blob/master/readme_graphics/fishkde.gif" width="800px" height="400px"></img>

## Classification Results


ROC & AUC            |  Confusion Matrix
:-------------------------:|:-------------------------:
<img src="https://github.com/joeshull/what_the_fish_beta/blob/master/readme_graphics/rocauc.png" width="600px" height="400px"></img>   |  <img src="https://github.com/joeshull/what_the_fish_beta/blob/master/readme_graphics/conf_matrix.png" width="600px" height="400px"></img>

ROC Curve



Confusion Matrix



Min/Max fish

<img src="https://github.com/joeshull/what_the_fish_beta/blob/master/readme_graphics/MaxMin.png" width="800px" height="600px"></img>

White_fish Classify

<img src="https://github.com/joeshull/what_the_fish_beta/blob/master/readme_graphics/not_fish.png" width="600px" height="400px"></img>


## Conclusion



