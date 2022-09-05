We see here some analyses of standard image pairs that are often used in published neural color transfer studies.

The images are ~700 x 500 pixels and 100% reshading is specified (for compatibility with other studies).

Baseline processing is Denoise as original processing.

Colour-channel 5-5-7 processing is the current default.  Blur colour channels at three different stages with blur factors 5, 5, 7.
(also there is one example of 5-5-7 to all channels).

All-channel-processing is a new option.  
Monochrome and the 2 colour channels are each seperately blurred with factors 1, 11,  11.
Blurring by 1 has no effect, so no blurring is applied to the Base image prior to colourgram processing. 
(The blur processing load is therefore reduced by one third).


For the Flower images we see that 1-11-11 is very similar to Baseline (and to the expected output), whereas 5-5-7 is not.  


The city images are instructive.

The 5-5-7 colour channel images show brown smearing to the right of the largest tower block but this effect is reduced if the greyshade/monochrome/luminance channel is also blurred. 

What is happening here is that the tower block is dark brown and has a brown component in the colour channels. If blur is applied only to the colour channels then the brown is spread but not the monochrome which masks it within the area of the tower block.

The 5-5-7 all channels does not show the brown fringe effect, because the monochrome is spread with the brown (to give a greyish fringe).  For this processing mode however, other brighter building do not look good and show brown patches.

The 1-11-11 image is crisp with the least colour distortion it is better than the original baseline image.

CONCLUSION: Use 1-11-11 all-channels as prefered processing but keep this choice under review.


NOTE:

The blurring of the monochrome (although now favoured) is more contentious for a number reasons.

In our processing, reassert processing reinforces the monochrome variation by re-applying a mix of Base image and/or Palette image luminosity (balance depends upon the shading paramter).  In doing so it overwrites any noise in the monochrome channel.

The human eye is less sensitive to the blur of colour than to the blur of  greyshade/monochrome/luminance.

(Slide 7 in the pdf file in this folder illustrates this.)


FROM 'VISION AND ART': THE BIOLOGY OF SEEING. MARGRET LIVINSTONE>
Our colour system operates at a surprisingly low resolution. That is cells that code colour have larger receptive fields and there are fewer of them than the cells used for resolving contours. This means that our perception of colour is coarse. In other words you don’t have to colour inside the lines. … Painters who use watercolours or pastels seem to have figured this out; they often exploit the low resolution of our colour system by applying their colour in a looser or blurrier way than the higher colour-contrast outlines of the objects the colour seems to conform.