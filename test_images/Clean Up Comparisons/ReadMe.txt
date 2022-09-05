
SORRY ORIGINAL NOTE BELOW PROBABLY WRONG.

BLUR PROBABLY WASN'T ORIGINALLY WORKING DUE TO WRONG SET UP.  SO MY 5-5-7 IMAGE IS PROBABLY NO SMOOTHING AT ALL.




ORIGINAL NOTE
I found an instance where using CV_blur was not as good as using CV_Denoise.

For CV_Blur I was twice pre-applying blur(5) on each of the base and palette colougrams and then twice applying blur(7) on the resultant image (call this 5-5-7). It can be seen that this is not as good as de-noise.  I overcame this by stepping up to (9-9-9) and now it looks OK.

Please check you agree with this.  We don't wish to commit to a particular understanding and then find later that is pointing us down the wrong path.