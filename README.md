# face-recognition
Face constuctor based on set ot different part of different human beings using OpenCV

It is construction the first set of constituent elements of the face, with which you can select the maximum copy of the input image of the face.

Person gender: Male (initial requirement)
Used format: PNG, resolution 200 x 200, monochrome image

Mechanism: Component groups used:

-Upper head (reverse "U-shaped"): Head contour, including hairstyle, if any: 30 elements
-The lower part of the head is "U-shaped":Chin contour, including beard if present: 30 elements
-Eyes: 40 elements
-Mouth: 30 elements
-Nose: 20 elements
-Eyebrows: 30 elements
-Mustache: 20 elements
-Ears: 5 elements

In each group, the first element is average over the entire group of emotionless components
