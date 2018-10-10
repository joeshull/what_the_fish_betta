# What The Fish (Beta) 
--- ***
An exploration in image classification using Logistic Regression

Tools: [Python , Numpy, Python Image Library, SKLearn]


## Why The Fish?
My friend likes to fish the rivers and lakes in Colorado, but often doesn't know what fish he's brought in. He asked for an ML tool that will identify fish species, but this is Capstone 1 so we're starting at the ground floor.


## Goals
Attempt a "fish or not-a-fish" classification using Logistic Regression using 33x33px grayscaled images on ~ 2000 pictures. 


## Image Acquisition

Downloaded ~1000 pictures for "fish" queried from Google Images

<div align="Center">
    <img src="https://github.com/joeshull/what_the_fish_beta/blob/master/readme_graphics/ex_fish.jpg" width="400px" height=200px"></img> 
</div>


Downloaded ~1000 pictures from 200 different "non-fish" categories

<div align="Center">
    <img src="https://github.com/joeshull/what_the_fish_beta/blob/master/readme_graphics/ex_bear.jpg" width="400px" height="350px"></img> 
</div>


## Image Processing

The images I downloaded started as RGB images in either JPEG or PNG form.

Here's an example of an RGB Photo zoomed in to the pixel level

<div align="Center">
    <img src="http://archive.xaraxone.com/webxealot/workbook35/rgb-cymk_04.gif" width="400px" height="400px"></img> 
</div>



courtesy of xaraxone.com

<div align="Center">
    <img src="http://archive.xaraxone.com/webxealot/workbook35/rgb-cymk_01.gif" width="400px" height="400px"></img> 
</div>
courtesy of xaraxone.com


I wanted to eliminate color as a variable to see if the classifier could simply identify some shape in the image. To do that I used PIL's "L" algoright to convert the image to grayscale and de facto provide us with a 2D matrix.

Greyscaled Using PIL's 'L' Algorithm
	L = R * 299/1000 + G * 587/1000 + B * 114/1000

<div align="Center">
    <img src="https://github.com/joeshull/what_the_fish_beta/blob/master/readme_graphics/gray_smile.png"></img> 
</div>

Fish Image                 |  Non-fish Image
:-------------------------:|:-------------------------:
<img src="https://github.com/joeshull/what_the_fish_beta/blob/master/readme_graphics/fish_proc.png" width="400px" height="400px"></img>   |  <img src="https://github.com/joeshull/what_the_fish_beta/blob/master/readme_graphics/nfish_proc.png" width="400px" height="400px"></img>


Logistic regression takes a 2d matrix as input, so each picture was flattened to a vector of length 1089 (33x33).


## Image EDA
<div align="Center">
	<img src="https://github.com/joeshull/what_the_fish_beta/blob/master/readme_graphics/avgimage.png" width="400px" height="200px"></img> 
</div>


Talk about Google Image Query Bias



-Graphic of Brightness distribution at a random pixel

<img src="https://github.com/joeshull/what_the_fish_beta/blob/master/readme_graphics/pixel_brightness.png" width="800px" height="400px"></img>
This probability distribution at each pixel will inform the classifier


<img src="https://github.com/joeshull/what_the_fish_beta/blob/master/readme_graphics/fishkde.gif" width="800px" height="400px"></img>

## Classification Results

ROC Curve
Confusion Matrix
Min/Max fish
White_fish Classify


## Conclusion



