# Optical_Tweezer
This project involves the segmentation and tracking of the optical beads under the microscope

- Step 1: Take in input all the video dataset and then process then process it frame by frame.



## General Instructions about the processing of the files...
- First load the variable `workspace.mat`
- Now refer the file `Optical_Tweezer_Code_Latest_v4.m` line 388 - 406
- 
- The input video from obtained from the EMCCD camera is of resolution 800 x1280 this is stored in the 
- Each frame is originally of resolution 800 x 1280, now we only want to crop it to 
- The original image obtained in the variable `frame` using the `readFrame()` function and then read into the framecrop variable by decimating rows: \n
- i.e. framecropped = frame(minY:maxY, minX:maxX)

`minY` = 277  <br>
`maxY` = 676  <br>
`minX` = 392  <br>
`maxX` = 816  <br>


- So the image dimension is  
= 676 - 277 + 1 = 400 for the rows
 and 
- 816 - 392 + 1 =  425 for the columns, 
- so the cropped image dimensions are 400 x 425 ( rows x columns), and this gives the cropped image.
- Total image dimension is 400 x 425.

## Preparation of the dataset

- Each cropped frame is then processed using the napari 
- `napari` is used to generate the masks directly and saved in the format as shown in the directory `cropped_dataset_final` with folders `images` and `masks` or visit the `final_tweezer_dataset` [https://www.kaggle.com/datasets/vinayakpathak312/final-tweezer-dataset]

- For reference you can also visit the Kaggle repo signed under BITS Username : f2014549@alumni.bits-pilani.ac.in[https://www.kaggle.com/code/vinayakpathak312/new-optical-tf/edit]


## Model architecture and code and training
- Model Description and architecture(to be added along with the image)
- Code for processing visit the link 
[https://www.kaggle.com/code/vinayakpathak312/new-optical-tf/edit]

<br>
# Appendix

- The code for resizing the input images and normalizing and changing the input images from uint16 to np.float32
Refer: Link [https://gist.github.com/vinayapathak/6db14b9fbc185a86a0675a38a98c0af5]
- The code for resizing the mask into the target size
Refer: Link [https://gist.github.com/vinayapathak/e6768de6f1edae4393476cb3e04bd1f4]