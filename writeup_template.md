# **Finding Lane Lines on the Road** 

## Writeup Template

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Pipeline.

I actually preferred to use a separate python script outside of the python notebook, and later I copied the script back
in the notebook.

The pipeline looks as follows:

0. Video is loaded
1. Grayscale
2. Blur
3. Find the edges with Canny
4. Setup a mask around the bottom part of the image
5. Find lines with HoughLines
6. Doing an average for the q and m parameters of all lines in order to reduce to 2 lines. Such average is weighted on
   the difference between the two y end points of the line to discard short and too horizontal lines. Also the final
   value is averaged with the latest 5 frames to soften the movement.
7. The lines are superimposed with the original image
8. Video is saved


### 2. Potential shortcomings

There could be an issue with turns as the lines would be way more horizontal and discarded by the weighted average

Another issue could arise from the car not being perfectly centered


### 3. Suggest possible improvements to your pipeline

The polygon for the mask should be dynamic and adapt based on the previous frame.

I'm sure the performances can be improved.