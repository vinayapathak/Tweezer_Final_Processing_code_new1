%% Tweezer processing code...

% Step-1: Write the code snippet for reading the .cine files converted into
% .tiff file 
% 
% Step-2: Run a sample snippet file, to calculate optical flow between two
% frames using in built MATLAB.
%
% Step-3(beta, later): Invoke Pytorch and Python code to improve the
% workflow
%
% Step-4: Analyse the workflow using graph, and describe the physical
% systems using model.
%
% Step-5: Explain the final results in terms of graphs etc.
%% Add neccesary paths 


clc
%addpath('proesmans-optical-flow-main')
%
parent_path = "../27808_1_3/";
addpath(parent_path)


%% Sample Optical Flow Calculation for a pair of image...

frame1 = imread('Img000740.tif');
frame2 = imread('Img000741.tif');

% Range of frames  = [700 - 740];

%% Normalize and Visualize the frames

% Visualize both the frames on top of each other.

figure;
subplot(2, 1, 1)
framenorm1 = normimg(frame1);
imshow(framenorm1)
title("frame1")
subplot(2, 1, 2)
framenorm2 = normimg(frame2);
imshow(framenorm2)
title("frame2")

%% Calculate the optical flow using the inbuilt MATLAB function

% Display the optical flow calculation from the two frames...

opticFlow = opticalFlowHS;


%% Create the GUI...
h = figure;
movegui(h);
hViewPanel = uipanel(h,'Position',[0 0 1 1],'Title','Plot of Optical Flow Vectors');
hPlot = axes(hViewPanel);



%% Estimate the optical flow from consecutive images...

flow = estimateFlow(opticFlow, framenorm1);
imshow(framenorm1)
hold on 
plot(flow)


%% Read the frames in the sequential mode and estimate the flow...

files = dir(fullfile(parent_path, '*.tif'));


%% Create figure handle for displaying result

h = figure;
movegui(h);
hViewPanel = uipanel(h,'Position',[0 0 1 1],'Title','Plot of Optical Flow Vectors');
hPlot = axes(hViewPanel);

%% Declare an Optical Flow object

opticFlow = opticalFlowLK('NoiseThreshold',0.009);

%% Estimate the optical flow for consecutive frames and display results
figure

%for i = 2:numel(files)
for i = 40:10:90
    
    % Normalize the original image
    frame1 = imread(fullfile(files(i).folder, files(i-1).name));
    framenorm1 = normimg(frame1);

    frame2 = imread(fullfile(files(i-1).folder, files(i).name));
    framenorm2 = normimg(frame2);
    
    
    % Estimate the optical flow for video frames
    %flow = estimateFlow(opticFlow, framenorm1, framenorm2);
    
    %Estimate the optical flow for an image pair not neccesarily a
    %consecutive video frame...

    flow = opticalFlow(framenorm1, framenorm2);

    % Visualize the optical flow overlayed on the original image.
    imagesc(framenorm2)
    colormap(gray)
    hold on
    plot(flow, 'DecimationFactor',[5, 5], 'ScaleFactor',10)
    title(i)
    hold off
    pause(10^-3)
   

end


%% Alternative code as compared to above..

% Start reading frame by frame from the video...

% First generate the .avi from the .tiffs


vidObj = VideoWriter('tweezer_video.avi'); %Instantiate the video object to start writing the frames into the video file

open(vidObj) %Open the video object to start recording the videos

for i  = 1:numel(files)

     curframe = normimg(imread(fullfile(files(i).folder, files(i).name)));  % Read frame by frame into the video object. Normalise the 16 bit image into the double.
     writeVideo(vidObj,curframe); %Write the current videoframe into a particualr videoObject

end


close(vidObj) % Closing the video object saves the video..



%% Estimate the Optical Flow from .avi, using Horn-Schunck

%% Main code starts here for now...

vidReader = VideoReader('tweezer_video.avi');
opticFlow = opticalFlowHS;

% h = figure;
% movegui(h);
% hViewPanel = uipanel(h,'Position',[0 0 1 1],'Title','Plot of Optical Flow Vectors');
% hPlot = axes(hViewPanel);

figure
while hasFrame(vidReader)
    frameRGB = readFrame(vidReader);
    frameGray = normimg(im2gray(frameRGB));  
    flow = estimateFlow(opticFlow,frameGray);
    imagesc(frameRGB)
    colormap("gray")
    hold on
    plot(flow,'DecimationFactor',[5 5],'ScaleFactor',6000);
    hold off
    %step(vidReader)
    pause(0.01)
end


%% The next step of steps : Tracking based flow estimation(no optical flow)

%1. Use the image segmenter to segment the original image containing the
%beads.
% a.) Go to Apps> Image processing and vision module >Click on Image
% Segmenter
% b.) Select find circles, set the following parameters 
% >> Min. Diameter 5 ; Max. Diameter 15; Number of Circles: 10; Foreground
% Polarity: `dark`; Senstivity: 0.7 - 0.9( you may try different values
% within this range)
% >> Click on `Find Circles`




% 2. Intialize the beads by finding and labelling the countours
% 3. Now for each contour use the Kalman or any other optical tracking
% algorithm for obtaining the results
% 4. Idea is to detect blobs in the images and get the track the blobs..





%% Display the original video

figure
video = VideoReader("tweezer_video.avi");
%open(video)
i = 0;
while hasFrame(video) && i<400

    frame = readFrame(video);
    imshow(frame);
    i = i+1;
end
%close(video)
%%  Now do the blob analysis for each frame...

% Step 1: Binarize each frame using the predefined code and visualize if the overlap...

figure;

video = VideoReader("tweezer_video.avi");
i = 0;

while hasFrame(video) && i<30

    frame = readFrame(video);

    subplot(3, 1, 1)
    imshow(frame);

    [BW,maskedImage] = segmentImage(im2gray(frame));

    subplot(3, 1, 2)
    imshow(BW)

    subplot(3, 1, 3)
    imshow(maskedImage)

    i = i+1;

end

%% Accuracy limited for radius <= 5 ; as shown by the warnings. Need to increase the local image size; would be helpful to find the circle radius...
figure;
imshow(frame)
%% This is to crop out frame for the time...
masimg = roipoly(frame) ;

% Find the vertices corners

[row, col] = find(masimg); %% Important function to find the corners of the binary mask
minX = min(col);
minY = min(row);
maxX = max(col);
maxY = max(row);



%% Now we try running the video for the cropped area and see if we could get satisfactory results for this cropped region

% Idea is to reduce time and computational expense

% Display the cropped image
figure;
imshow(frame(minY:maxY, minX:maxX))
%% Display cropped video to ensure the tweezer movement is encompassed within the Field of View(FOV)

figure;

video = VideoReader("tweezer_video.avi");
i = 0;

%while hasFrame(video) && i<500 %numel(

while hasFrame(video)

    frame = readFrame(video);
    framecrop = frame(minY:maxY, minX:maxX);

    subplot(3, 1, 1)
    imshow(framecrop);

    [BW,maskedImage] = segmentImage(im2gray(framecrop));

    subplot(3, 1, 2)
    imshow(BW)

    subplot(3, 1, 3)
    imshow(maskedImage)

    i = i+1;

    pause(0.001)

end


%%
%% Display cropped video to ensure the tweezer movement is encompassed within the Field of View(FOV) - bright polarity

figure;

video = VideoReader("tweezer_video.avi");
i = 0;

%while hasFrame(video) && i<500 %numel(

while hasFrame(video) && numel(files)

    frame = readFrame(video);
    framecrop = frame(minY:maxY, minX:maxX);

    subplot(3, 1, 1)
    imshow(framecrop);

    [BW,maskedImage] = segmentImagebright(im2gray(framecrop));

    subplot(3, 1, 2)
    imshow(BW)

    subplot(3, 1, 3)
    imshow(maskedImage)

    i = i+1;

    pause(0.001)

end


%% Use the segmenter on the contrasted images








%% Circles are hard to detect using MATLAB inbuilt image segmenter; so right now invoking the inbuilt blob analyzer..

figure;
%Link : https://in.mathworks.com/help/releases/R2021a/vision/ug/cell-counting.html
video = VideoReader("tweezer_video.avi");

i = 0;


hblob = vision.BlobAnalysis( ...
                'AreaOutputPort', false, ...
                'BoundingBoxOutputPort', false, ...
                'OutputDataType', 'single', ...
                'MinimumBlobArea', 50, ...
                'MaximumBlobArea', 300, ...
                'MaximumCount', 60);

% Acknowledgement
ackText = ['Data set courtesy of Krishna Kant Singh, FemtoLab' ...
             'Indian Institute of Technology, Kanpur'];


%The blob analyzer can just be used to find the centroid locations of the
%existing blobs; the links are mentioned above....

%% To binarize can try imbinarize function

figure;
imshow(frame)

framethresholded = imbinarize(rgb2gray(frame));
figure;
imshow(framethresholded)


%%


%% To adjust the adaptive histpgram...
figure;
imshow(adapthisteq(frameGray));  %% This is just improving the contrast sing the adpative histogram equalizaation

%% %% --> Cur code:Apply the binarization on this histogram equalized dataset...
figure;

video = VideoReader("tweezer_video.avi");


startFrame = 700;
i = startFrame;

video.CurrentTime = (startFrame - 1)/video.FrameRate;

%while hasFrame(video) && i<500 %numel(

while hasFrame(video)% && numel(files)  % Start the Video object visualization 



    frame = readFrame(video);
    framecrop = frame(minY:maxY, minX:maxX);


    subplot(3, 1, 1)
    imshow(im2gray(framecrop));

    %[BW,maskedImage] = segmentImageparams(adapthisteq(im2gray(framecrop))); % With contrast adjustment
    [BW,maskedImage] = seg;ibrssscsssvp`
    mentImageparams(im2gray(framecrop), 0.85, 'dark', 6, 30);

    subplot(3, 1, 2)
    imshow(BW)

    subplot(3, 1, 3)
    imshow(maskedImage)

    i = i+1;

    pause(0.001)

   

   titlenam = sprintf(strcat("Frame = ", num2str(i), "; Time = ", num2str(video.CurrentTime)));
   sgtitle(titlenam)% Adding super title for better visualization

end

%release(video)

%%  Final Code to find the circles in the low contrast region ( when beads goes out of focus)


figure;

video = VideoReader("tweezer_video.avi");


startFrame = 200;
i = startFrame;
endFrame = 900;

video.CurrentTime = (startFrame - 1)/video.FrameRate;

%while hasFrame(video) && i<500 %numel(

while hasFrame(video) && i<endFrame  % Start the Video object visualization at a given frame and limit before a particular number



    frame = readFrame(video);
    framecrop = frame(minY:maxY, minX:maxX);


    subplot(3, 1, 1)
    imshow(im2gray(adapthisteq(framecrop)));

    %[BW,maskedImage] = segmentImageparams(adapthisteq(im2gray(framecrop))); % With contrast adjustment
    [BW,maskedImage] = segmentImageparams(im2gray(framecrop), 0.85, 'dark', 8, 100);

    subplot(3, 1, 2)
    imshow(BW)

    subplot(3, 1, 3)
    imshow(maskedImage)
    
    i = i+1;
   
    pause(0.001)
   titlenam = sprintf(strcat("Frame = ", num2str(i), "; Time = ", num2str(video.CurrentTime)));
   sgtitle(titlenam)% Adding super title for better visualization

end

%release(video)

%% Solving the issues at top right corner 
%%  Final Code to find the circles in the low contrast region ( when beads goes out of focus)


figure;

video = VideoReader("tweezer_video.avi");


startFrame = 320;
i = startFrame;
endFrame = 321;

video.CurrentTime = (startFrame - 1)/video.FrameRate;

%while hasFrame(video) && i<500 %numel(

while hasFrame(video) && i<endFrame  % Start the Video object visualization at a given frame and limit before a particular number

    frame = readFrame(video);
    %framecrop = frame(minY:maxY, minX:maxX);
    
    I = im2gray(frame);

    subplot(2, 1, 1)
    imshow(I);

    %[BW,maskedImage] = segmentImageparams(adapthisteq(im2gray(framecrop))); % With contrast adjustment
    %[BW,maskedImage] = segmentImageparams(im2gray(frame), 0.85, 'dark', 8, 100);

    subplot(2, 1, 2)
    %[Gmag, Gdir] = edge(I,'canny');
    %imshow(Gmag)
    imagesc(normimg(adapthisteq(imgaussfilt(I, 2)-I)));
    colormap('gray')
    i = i+1;
   
    pause(0.001)
   titlenam = sprintf(strcat("Frame = ", num2str(i), "; Time = ", num2str(video.CurrentTime)));
   sgtitle(titlenam)% Adding super title for better visualization

end

framecopy = frame;
%% Cropped Segmentation
figure;
crop = 'y';
if crop=='y'
    %framecopy = frame;
    %frame = frame(300:400,600:800);
    %frame = frame(418:520, 480:655);% for other beads
    %frame = edge(frame(449:510, 528:586), 'prewitt');% for other beads
    frame = frame(449:510, 528:586);
else
    frame = frame;
end
[BW,maskedImage] = segmentImageparams(im2gray(frame), 0.9, 'dark', 8, 100);
subplot(3, 1, 1)
imshow(im2gray(frame))

subplot(3, 1, 2)
imshow(BW)
subplot(3, 1, 3)
imshow(maskedImage)
frame = framecopy;
%% Solving the bead intertwining problem


% Enhance contrast (you may need to adjust parameters)
enhanced_image = adapthisteq(frametest);

figure;imshow(enhanced_image)

%%
% Choose a threshold value (you may need to adjust this)
threshold =  graythresh(enhanced_image);

%%


binary_image = imbinarize(enhanced_image, threshold);
%

%% Display the segmented region
% se = strel('disk', 1); % Define a structuring element
% binary_image = imopen(binary_image, se);
figure(11);imshow(imfill(binary_image ,[2 2], 26));





%% Current Aims on improving binary thresholding

% First try the sliding window approach

% Define the ROI on original frame
% Define the sliding window size and the strides (we will take the full
% slides approach).
% Find the circles or blobs in the images ( Take union of circles and blobs)


figure; imshow(framecopy)

% roi = roipoly(im2gray(framecopy));
%% Found the cropped region

[row, col] = find(roi);
minx = min(col);
miny = min(row);
maxx= max(col);
maxy = max(row);


%% Decide the size of the window
figure(998)
window = roipoly(framecopy);


%%
figure;imshow(window)

%% Run the video

%%  Final Code to find the circles in the low contrast region ( when beads goes out of focus)


figure;

video = VideoReader("tweezer_video.avi");


startFrame = 341;
i = startFrame;
endFrame = 342;

video.CurrentTime = (startFrame - 1)/video.FrameRate;

%while hasFrame(video) && i<500 %numel(

while hasFrame(video) && i<endFrame  % Start the Video object visualization at a given frame and limit before a particular number

    frame = readFrame(video);
    %framecrop = frame(minY:maxY, minX:maxX);
    
    I = im2gray(frame);

    subplot(2, 1, 1)
    imshow(I);

    %[BW,maskedImage] = segmentImageparams(adapthisteq(im2gray(framecrop))); % With contrast adjustment
    %[BW,maskedImage] = segmentImageparams(im2gray(frame), 0.85, 'dark', 8, 100);

    subplot(2, 1, 2)
    %[Gmag, Gdir] = edge(I,'canny');
    %imshow(Gmag)
    imagesc(normimg(adapthisteq(imgaussfilt(I, 2)-I)));
    colormap('gray')
    i = i+1;
   
    pause(0.001)
   titlenam = sprintf(strcat("Frame = ", num2str(i), "; Time = ", num2str(video.CurrentTime)));
   sgtitle(titlenam)% Adding super title for better visualization

end

framecopy = frame;
%% Define window limit...

[windowrow, windowcol] =find(window);
minwinx = min(windowcol);
minwiny = min(windowrow);
maxwinx = max(windowcol);
maxwiny = max(windowrow);
winwidth = maxwinx - minwinx;
winheight = maxwiny - minwiny;

stride = 1;

rownum = round(size(framecopy, 1)./winheight);
colnum = round(size(framecopy, 2)./winwidth);
stridex = round(winwidth*1);
stridey = round(winheight*1);

%% Display the bounding box across entire image
for i = 1:stridex:size(framecopy, 2)- winwidth - 2
    for j = 1:stridey:size(framecopy, 1) - winheight - 2
        figure(12);
        subplot(2, 1, 1)
        imshow(framecopy)
        rectangle('Position', [i, j, winwidth, winheight], 'EdgeColor', 'r', 'LineWidth', 2);
        subplot(2, 1, 2)
        imshow(framecopy(j:j+winheight, i:i+winwidth));
        pause(0.0001)
    end
end
%% Now let's produce the binary mask across the entire image by limiting the FOV.
for i = minx:stridex:maxx
    for j = miny:stridey:maxy
        figure(12);
        subplot(2, 1, 1)
        imshow(framecopy)
        rectangle('Position', [i, j, winwidth, winheight], 'EdgeColor', 'r', 'LineWidth', 2);
        subplot(2, 1, 2)
        imshow(framecopy(j:j+winheight, i:i+winwidth));
        pause(0.0001)
    end
end

%% Now we have to binarize

mask = zeros(size(framecopy, 1), size(framecopy, 2));

Ihist = adapthisteq(im2gray(framecopy));

for i = minx:stridex:maxx
    for j = miny:stridey:maxy
        figure(12);
        subplot(2, 1, 1)
        imshow(framecopy)
        rectangle('Position', [i, j, winwidth, winheight], 'EdgeColor', 'r', 'LineWidth', 2);
        subplot(2, 1, 2)
        Itemp = Ihist(j:j+winheight, i:i+winwidth);
        imshow(Itemp);

        %subplot(3, 1, 3)
        maskcopy = segmentImageparams(Itemp, 0.888, 'dark', 6, 60);
        mask(j:j+winheight, i:i+winwidth) = maskcopy;
        pause(0.0001)
    end
    
end


%% Fit circles to boundaries in an image..

binaryImage = iframemaskcur;

boundaries = bwboundaries(binaryImage);

% Create a figure and axis for plotting
figure;

% Display the binary image using 'b' option to create a black and white display
imshow(binaryImage, 'InitialMagnification', 'fit');
hold on;
for k = 1:length(boundaries)
    boundary = boundaries{k};
    plot(boundary(:, 2), boundary(:, 1), 'r', 'LineWidth', 2); % 'r' for red color, adjust line properties as needed
    % Calculate the area of a specific area;
    area = polyarea(boundary(:, 2), boundary(:, 1));
    round([k, area])
end




%%  Constrict FOV before processing...

BWconstrict = roipoly(iframemaskcur);


%% % Sample code not part of main code--to save the cropped images...

video = VideoReader("tweezer_video.avi");


startFrame = 20;
video.CurrentTime = (startFrame - 1)/video.FrameRate;
endFrame = 820;

i = 0;

while hasFrame(video) && i<endFrame  % Start the Video object visualization at a given frame and limit before a particular number

    frame = readFrame(video);
    croppedFrame = frame(minY:maxY, minX:maxX);
    filename = ["cropped_dataset/" sprintf('%05d.tiff', i)];
    filenamef = strcat(filename(1),filename(2));
    imwrite(imresize(croppedFrame, [512,512]),filenamef)
    i = i + 1;
 
end






%% Current Code(04-11-23-02:24am): %% More integrated version Kalman filtering lesser noisy version by using time based auto correlation

% Use adapytive version of imbinarize
%https://in.mathworks.com/help/images/ref/imbinarize.html



% Integrate the kalman filtering into the code...



video = VideoReader("tweezer_video.avi");


startFrame = 300;%%5;%6;%316;%320;%341
i = startFrame;
endFrame = 820;%550;%numel(files);%342;
c = 0;

video.CurrentTime = (startFrame - 1)/video.FrameRate;

%while hasFrame(video) && i<500 %numel(
 % Dilation window size
winsize = 8;
SE = strel("rectangle",[winsize winsize]);
SER = strel("disk", 5);
SERnew = strel("disk", 5);
initialFrame = readFrame(video);
initialFrame = initialFrame(minY:maxY, minX:maxX, :);
%initalFramemask = segmentImageparams(adapthisteq(im2gray(initialFrame)), 0.85, 'dark', 8, 100);
%initialFramemask = imdilate(1-imbinarize(adapthisteq(im2gray(initialFrame))), SE);

initialFramemask = imdilate(1-imbinarize(adapthisteq(im2gray(initialFrame)), "adaptive", 'ForegroundPolarity','dark','Sensitivity',0.4), SE);
ot.frame = adapthisteq(im2gray(initialFrame));

%  Setup the Kalman Filtering. ot structure...(Optical Tracking for Optical Tweezers)

ot.tracks = initializeTracks(ot);  % Create empty tracks..
ot.centroids = [];
ot.bboxes = [];
ot.mask = [];
ot.nextId = 1; % ID of the next track
ot.blobAnalyser = vision.BlobAnalysis('BoundingBoxOutputPort', true, ...
            'AreaOutputPort', true, 'CentroidOutputPort', true, ...
            'MinimumBlobArea', 30, ...
            'MaximumCount', 8);
% ot.blobAnalyser= vision.BlobAnalysis( ...
%             'AreaOutputPort', true, ...
%             'BoundingBoxOutputPort', true, ...
%             'OutputDataType', 'single', ...
%             'MinimumBlobArea', 50, ...
%             'MaximumBlobArea', 300, ...
%             'MaximumCount', 60);
invisibleForTooLong = 40;
ageThreshold = 3;
visibilityThresh = 0.6;





%
%figure(100);
while hasFrame(video) && i<endFrame  % Start the Video object visualization at a given frame and limit before a particular number
    
    frame = readFrame(video);
    framecrop = frame(minY:maxY, minX:maxX);
    
    I = adapthisteq(im2gray(framecrop));
    ot.frame = I;
    
    %%subplot(3, 2, 1)
    %%imshow(I);
    
    %[BW,maskedImage] = segmentImageparams(adapthisteq(im2gray(framecrop))); % With contrast adjustment
    %[BW,maskedImage] = segmentImageparams(im2gray(frame), 0.85, 'dark', 8, 100);
    
    %%subplot(3, 2, 2)
    %[Gmag, Gdir] = edge(I,'canny');
    %imshow(Gmag)
    %Iframetemp = imdilate(1-imbinarize(I), SE);
    Iframetemp = 1-imbinarize(I);
    %mask = segmentImageparams(I, 0.9, 'dark', 8, 100);
    %imshow(1-imbinarize(I));
    %imshow(imdilate(1-imbinarize(I), SER))
    itempcorr = and(Iframetemp, initialFramemask);
    %imshow(bwmorph(imdilate(itempcorr, SERnew), 'clean'))
    SERlatest = strel("disk", 5);
    SERlatestclosesmall = strel("disk", 5);
    conn = 8;
    
   % iframemaskcur = (1-BWconstrict).*foregroundmask.*clean_erode_dilate(bwmorph(bwmorph(bwmorph(itempcorr, 'close'), 'thicken'), 'fill'));

    iframemaskcur = (1-BWconstrict).*foregroundmask.*clean_erode_dilate(itempcorr);
    
    iframemaskcur = imfill(iframemaskcur, 'holes');
    iframemaskcur = imclose(iframemaskcur, SERlatestclosesmall);
    iframemaskcur = imerode(iframemaskcur, SERlatest);

    if c == 0
        %imshow(iframemaskcur)
    else
        %imshow(and(iframemasknext,iframemaskcur))
    end
    %imshow(itempcorr)
    iframemasknext = iframemaskcur;
    %imshow(bwulterode(imdilate(1-imbinarize(I), SE)));
    %imshow(bwmorph(imdilate(1-imbinarize(I), SE), 'remove'))
    %Iframetemp
    %imshow(segmentImageparams(Iframetemp, 0.85, 'dark', 8, 100));
    %if i>314
    %initialFramemask =imdilate(1-imbinarize(I), SE);
    initialFramemask =1-imbinarize(I);
    
    %%subplot(3, 2, 3)


    maskcur = segmentImageparams(I, 0.85, 'dark', 8, 100);
    %imshow(maskcur)
    title('mask')

    %%subplot(3, 2, 4)
    
    %imshow(xor(maskcur, itempcorr))
    title('mask')


    pause(0.001)
    titlenam = sprintf(strcat("Frame = ", num2str(i), "; Time = ", num2str(video.CurrentTime)));
    sgtitle(titlenam)% Adding super title for better visualization


    % Kalman based Filtering...

    [ot.centroids, ot.bboxes, ot.mask] = detectObjects(iframemaskcur, ot); % Fixed

    predictNewLocationsOfTracks(ot); %Fixed

    [assignments, unassignedTracks, unassignedDetections] = ...
        detectionToTrackAssignment(ot); %Fixed

    updateAssignedTracks(ot); %Fixed

    updateUnassignedTracks(ot); %Fixed

    deleteLostTracks(ot, invisibleForTooLong, ageThreshold, visibilityThresh); % Fixed

    createNewTracks(ot);

    displayTrackingResults(ot);





    i = i+1;
    c = c+1;


end

framecopy = frame;


%% Current Code: Augmented tracking - Current Code( 04-11-23-07:18 am )
% Currently the plan is to apply binarization based tracking on the top
% right corner. And apply the tracking on bottom top using findcircles.
% Let us apply this collectively

% Create the foreground mask
BWconstrict = roipoly(initialFrame);

%% Remove the dust particle ( mask creation)

dustmask = roipoly(initialFrame);

%% Additional constraint mask if required... (Blended mask)...
figure
% Use adapytive version of imbinarize
%https://in.mathworks.com/help/images/ref/imbinarize.html
% Integrate the kalman filtering into the code...
video = VideoReader("tweezer_video.avi");
startFrame = 320;%%5;%6;%316;%320;%341
i = startFrame;
endFrame = 776;%550;%numel(files);%342;
c = 0;
video.CurrentTime = (startFrame - 1)/video.FrameRate;
%while hasFrame(video) && i<500 %numel(
 % Dilation window size
winsize = 8;
SE = strel("rectangle",[winsize winsize]);
SER = strel("disk", 5);
SERnew = strel("disk", 5);
initialFrame = readFrame(video);
initialFrame = initialFrame(minY:maxY, minX:maxX, :);
%initalFramemask = segmentImageparams(adapthisteq(im2gray(initialFrame)), 0.85, 'dark', 8, 100);
%initialFramemask = imdilate(1-imbinarize(adapthisteq(im2gray(initialFrame))), SE);
%initialFramemask = imdilate(1-imbinarize(adapthisteq(im2gray(initialFrame)), "adaptive", 'ForegroundPolarity','dark','Sensitivity',0.4), SE);
intialFramemask = imclose(imfill(adapthisteq(im2gray(initialFramemask)),8, 'holes'), SER);
%intialFramemask = segmentImageparams()

ot.frame = adapthisteq(im2gray(initialFrame));
%  Setup the Kalman Filtering. ot structure...(Optical Tracking for Optical Tweezers)
ot.tracks = initializeTracks(ot);  % Create empty tracks..
ot.centroids = [];
ot.bboxes = [];
ot.mask = [];
ot.nextId = 1; % ID of the next track
ot.blobAnalyser = vision.BlobAnalysis('BoundingBoxOutputPort', true, ...
            'AreaOutputPort', true, 'CentroidOutputPort', true, ...
            'MinimumBlobArea', 30, ...
            'MaximumCount', 8);
% ot.blobAnalyser= vision.BlobAnalysis( ...
%             'AreaOutputPort', true, ...
%             'BoundingBoxOutputPort', true, ...
%             'OutputDataType', 'single', ...
%             'MinimumBlobArea', 50, ...
%             'MaximumBlobArea', 300, ...
%             'MaximumCount', 60);
invisibleForTooLong = 40;
ageThreshold = 3;
visibilityThresh = 0.8;

while hasFrame(video) && i<endFrame  % Start the Video object visualization at a given frame and limit before a particular number
    
    frame = readFrame(video);
    framecrop = frame(minY:maxY, minX:maxX);
    
    I = adapthisteq(im2gray(framecrop));
    ot.frame = I;
    
    %%subplot(3, 2, 1)
    %%imshow(I);
    
    %[BW,maskedImage] = segmentImageparams(adapthisteq(im2gray(framecrop))); % With contrast adjustment
    %[BW,maskedImage] = segmentImageparams(im2gray(frame), 0.85, 'dark', 8, 100);
    
    %%subplot(3, 2, 2)
    %[Gmag, Gdir] = edge(I,'canny');
    %imshow(Gmag)
    %Iframetemp = imdilate(1-imbinarize(I), SE);
    %Iframetemp = 1-imbinarize(I);
    Iframetemp = segmentImageparams(I, 0.9, 'dark', 8, 100);
    %mask = segmentImageparams(I, 0.9, 'dark', 8, 100);
    %imshow(1-imbinarize(I));
    %imshow(imdilate(1-imbinarize(I), SER))
    itempcorr = and(Iframetemp, initialFramemask);
    %imshow(bwmorph(imdilate(itempcorr, SERnew), 'clean'))
    SERlatest = strel("disk", 5);
    SERlatestclosesmall = strel("disk", 5);
    conn = 8;
    
    iframemaskcur = (BWconstrict).*foregroundmask.*clean_erode_dilate(bwmorph(bwmorph(bwmorph(itempcorr, 'close'), 'thicken'), 'fill'));

    %iframemaskcur = (BWconstrict).*foregroundmask.*clean_erode_dilate(itempcorr);
    
    iframemaskcur = imfill(iframemaskcur, 'holes');
    %iframemaskcur = imclose(iframemaskcur, SERlatestclosesmall);
    iframemaskcur = imerode(iframemaskcur, SERlatest);

    if c == 0
        %imshow(iframemaskcur)
    else
        %imshow(and(iframemasknext,iframemaskcur))
    end
    %imshow(itempcorr)
    iframemasknext = iframemaskcur;
    %imshow(bwulterode(imdilate(1-imbinarize(I), SE)));
    %imshow(bwmorph(imdilate(1-imbinarize(I), SE), 'remove'))
    %Iframetemp
    %imshow(segmentImageparams(Iframetemp, 0.85, 'dark', 8, 100));
    %if i>314
    %initialFramemask =imdilate(1-imbinarize(I), SE);
    initialFramemask =1-imbinarize(I);
    
    %%subplot(3, 2, 3)


    maskcur = segmentImageparams(I, 0.85, 'dark', 8, 100);
    %imshow(maskcur)
    title('mask')

    %%subplot(3, 2, 4)
    
    %imshow(xor(maskcur, itempcorr))
    title('mask')


    pause(0.001)
    titlenam = sprintf(strcat("Frame = ", num2str(i), "; Time = ", num2str(video.CurrentTime)));
    sgtitle(titlenam)% Adding super title for better visualization


    % Kalman based Filtering...

    [ot.centroids, ot.bboxes, ot.mask] = detectObjects(iframemaskcur, ot); % Fixed

    predictNewLocationsOfTracks(ot); %Fixed

    [assignments, unassignedTracks, unassignedDetections] = ...
        detectionToTrackAssignment(ot); %Fixed

    updateAssignedTracks(ot); %Fixed

    updateUnassignedTracks(ot); %Fixed

    deleteLostTracks(ot, invisibleForTooLong, ageThreshold, visibilityThresh); % Fixed

    createNewTracks(ot); % Fixed

    displayTrackingResults(ot);

    i = i+1;
    c = c+1;


end


%% Binary mask for the bead..

figure;
lowcontrastmask = roipoly(I);

%% 

figure;imshow((1-dustmask))
%% Current code 06-11-2023 - 18:17 Tracking with blending (without direclty applying masks)...

% Additional constraint mask
figure
% Use adapytive version of imbinarize
%https://in.mathworks.com/help/images/ref/imbinarize.html
% Integrate the kalman filtering into the code...
video = VideoReader("tweezer_video.avi");
startFrame = 5;%%5;%6;%316;%320;%341
i = startFrame;
endFrame = 776;%550;%numel(files);%342;
c = 0;
video.CurrentTime = (startFrame - 1)/video.FrameRate;
%while hasFrame(video) && i<500 %numel(
 % Dilation window size
winsize = 8;
SE = strel("rectangle",[winsize winsize]);
SER = strel("disk", 5);
SERnew = strel("disk", 5);
initialFrame = readFrame(video);
initialFrame = initialFrame(minY:maxY, minX:maxX, :);
%initalFramemask = segmentImageparams(adapthisteq(im2gray(initialFrame)), 0.85, 'dark', 8, 100);
%initialFramemask = imdilate(1-imbinarize(adapthisteq(im2gray(initialFrame))), SE);
%initialFramemask = imdilate(1-imbinarize(adapthisteq(im2gray(initialFrame)), "adaptive", 'ForegroundPolarity','dark','Sensitivity',0.4), SE);
intialFramemask = imclose(imfill(adapthisteq(im2gray(initialFramemask)),8, 'holes'), SER);  % Here 
%intialFramemask = segmentImageparams()

ot.frame = adapthisteq(im2gray(initialFrame));
%  Setup the Kalman Filtering. ot structure...(Optical Tracking for Optical Tweezers)
ot.tracks = initializeTracks(ot);  % Create empty tracks..
ot.centroids = [];
ot.bboxes = [];
ot.mask = [];
ot.nextId = 1; % ID of the next track
ot.blobAnalyser = vision.BlobAnalysis('BoundingBoxOutputPort', true, ...
            'AreaOutputPort', true, 'CentroidOutputPort', true, ...
            'MinimumBlobArea', 30, ...
            'MaximumCount', 8);
% ot.blobAnalyser= vision.BlobAnalysis( ...
%             'AreaOutputPort', true, ...
%             'BoundingBoxOutputPort', true, ...
%             'OutputDataType', 'single', ...
%             'MinimumBlobArea', 50, ...
%             'MaximumBlobArea', 300, ...
%             'MaximumCount', 60);
invisibleForTooLong = 40;
ageThreshold = 3;
visibilityThresh = 0.8;

while hasFrame(video) && i<endFrame  % Start the Video object visualization at a given frame and limit before a particular number
    
    frame = readFrame(video);
    framecrop = frame(minY:maxY, minX:maxX);
    I = adapthisteq(im2gray(framecrop));
    ot.frame = I;
    %%subplot(3, 2, 1)
    %%imshow(I);
    %[BW,maskedImage] = segmentImageparams(adapthisteq(im2gray(framecrop))); % With contrast adjustment
    %[BW,maskedImage] = segmentImageparams(im2gray(frame), 0.85, 'dark', 8, 100);
    %%subplot(3, 2, 2)
    %[Gmag, Gdir] = edge(I,'canny');
    %imshow(Gmag)
    %Iframetemp = imdilate(1-imbinarize(I), SE);
    %Iframetemp = 1-imbinarize(I);
    Iframetemp = segmentImageparams(I, 0.9, 'dark', 8, 100);

    %% Binarize the image selectrively by windows...

    window_size = [50, 50];

    



    %%
    %mask = segmentImageparams(I, 0.9, 'dark', 8, 100);
    %imshow(1-imbinarize(I));
    %imshow(imdilate(1-imbinarize(I), SER))
    itempcorr = and(Iframetemp, initialFramemask);
    %imshow(bwmorph(imdilate(itempcorr, SERnew), 'clean'))
    SERlatest = strel("disk", 5);
    SERlatestclosesmall = strel("disk", 5);
    conn = 8;
    iframemaskcur = (BWconstrict).*foregroundmask.*clean_erode_dilate(bwmorph(bwmorph(bwmorph(itempcorr, 'close'), 'thicken'), 'fill'));
    %iframemaskcur = (BWconstrict).*foregroundmask.*clean_erode_dilate(itempcorr);
    iframemaskcur = imfill(iframemaskcur, 'holes');
    %iframemaskcur = imclose(iframemaskcur, SERlatestclosesmall);
    iframemaskcur = (1-dustmask).*imerode(iframemaskcur, SERlatest); % Remove the dust particle...
    

    % Obtain labeled version of the mask...

    labeled_mask = bwlabel(iframemaskcur);
    
    % Obtain all the properties for each region...

    srbndall = regionprops(labeled_mask, "all"); % Finding the properties of all the connected regions.
    
    % %     Ony uncomment if you need to prepare the table.
    
    %     azprop = zeros(numel(srbndall),numel(srbndall)); % azprop stores the properties of each regions(Circularity and Area)
    % 
    %     for i  = 1:numel(srbndall)
    % 	    azprop(i, 1) = srbndall(i).Circularity ;
    %         azprop(i, 2) = srbndall(i).Area;
    %     end
    
        
    % Selectively erode the regions of the binary mask...
    % Define a threshold for region area
    min_area_threshold = 500;
    
    % Initialize a mask for selective erosion
    eroded_mask = zeros(size(iframemaskcur));
    
    % Iterate through each region and erode if area is too large
    for i = 1:length(srbndall)
        if srbndall(i).Area >= min_area_threshold
            region = ismember(labeled_mask, i);
            eroded_region = imerode(region, strel('disk', 3));  % You can choose the structuring element size as needed
            eroded_mask(eroded_region) = 0;
        end
    end
    
    % Display the original mask and the eroded mask
    figure(10)
    
    subplot(2, 1, 1)
    imshow(iframemaskcur);
    
    subplot(2, 1, 2)
    imshow(eroded_mask);
    % Apply watershed transform...
    % Calculate watershed transform, to distinguish the overlapping blobs.
    L = round(bwdist(eroded_mask));
     figure;imshow(L, [])
    %
    rgb = label2rgb(L,'jet',[.5 .5 .5]);
    % surf(rgb, 'EdgeColor', 'none');
    % colormap(jet);
    % view(3);
    title('Watershed Transform');
    a1 = imregionalmin(rgb2gray(rgb), 4);
    a2 = imregionalmax(rgb2gray(rgb), 4);

    se = strel("disk", 3);
 
    iframemaskcur = L;%bwmorph(imerode(1 - xor(a1, a2), se), "hbreak");

    %figure;imshow(iframemaskcur)
    %
    if c == 0
        %imshow(iframemaskcur)
    else
        %imshow(and(iframemasknext,iframemaskcur))
    end
    %imshow(itempcorr)
    iframemasknext = iframemaskcur;
    %imshow(bwulterode(imdilate(1-imbinarize(I), SE)));
    %imshow(bwmorph(imdilate(1-imbinarize(I), SE), 'remove'))
    %Iframetemp
    %imshow(segmentImageparams(Iframetemp, 0.85, 'dark', 8, 100));
    %if i>314
    %initialFramemask =imdilate(1-imbinarize(I), SE);
    initialFramemask =1-imbinarize(I);
    %%subplot(3, 2, 3)
    %  maskcur = segmentImageparams(I, 0.85, 'dark', 8, 100);
    %imshow(maskcur)
    title('mask')
    %%subplot(3, 2, 4)
    %imshow(xor(maskcur, itempcorr))
    title('mask')
    pause(0.001)
    titlenam = sprintf(strcat("Frame = ", num2str(i), "; Time = ", num2str(video.CurrentTime)));
    sgtitle(titlenam)% Adding super title for better visualization
    
    
    % Kalman based Filtering...
    [ot.centroids, ot.bboxes, ot.mask] = detectObjects(iframemaskcur, ot); % Fixed
    predictNewLocationsOfTracks(ot); %Fixed
    [assignments, unassignedTracks, unassignedDetections] = ...
        detectionToTrackAssignment(ot); %Fixed
    updateAssignedTracks(ot); %Fixed
    updateUnassignedTracks(ot); %Fixed
    deleteLostTracks(ot, invisibleForTooLong, ageThreshold, visibilityThresh); % Fixed
    createNewTracks(ot); % Fixed
    displayTrackingResults(ot);
    i = i+1;
    c = c+1;
end

%% Functions:

% Functions for the optical tracking...
%Create System Objects
%Create System objects used for reading the video frames, detecting foreground objects, and displaying results.


%Track initialization
  function tracks = initializeTracks(ot)
        % create an empty array of tracks
        tracks = struct(...
            'id', {}, ...
            'bbox', {}, ...
            'kalmanFilter', {}, ...
            'age', {}, ...
            'totalVisibleCount', {}, ...
            'consecutiveInvisibleCount', {});
    end


% Detect the objects in the given frame...

   function [centroids, bboxes, mask] = detectObjects(frame, ot)

%         % Detect foreground.
%         mask = obj.detector.step(frame);
% 
%         % Apply morphological operations to remove noise and fill in holes.
%         mask = imopen(mask, strel('rectangle', [3,3]));
%         mask = imclose(mask, strel('rectangle', [15, 15]));
%         mask = imfill(mask, 'holes');
% 
%         % Perform blob analysis to find connected components.
        
       


       % mask = segmentImageparams(frame, 0.85, 'dark', 6, 23);

     %   [~, centroids, bboxes] = ot.blobAnalyser.step(mask);
        [~, centroids, bboxes] = ot.blobAnalyser.step(logical(frame));
        mask = frame;
        assignin('base', 'ot', ot);
   end



% Predict the new locations of the existing tracks...
% Use the Kalman filter to predict the centroid of each track in the current frame, and update its bounding box accordingly.
function  predictNewLocationsOfTracks(ot)
        for i = 1:length(ot.tracks)
            bbox = ot.tracks(i).bbox;

            % Predict the current location of the track.
            predictedCentroid = predict(ot.tracks(i).kalmanFilter);

            % Shift the bounding box so that its center is at
            % the predicted location.
            predictedCentroid = int32(predictedCentroid) - bbox(3:4) / 2;
            ot.tracks(i).bbox = [predictedCentroid, bbox(3:4)];
        end

        assignin('base', 'ot', ot);

end

%% Assign Detections to Tracks

%Assigning object detections in the current frame to existing tracks is done by minimizing cost. 
%The cost is defined as the negative log-likelihood of a detection corresponding to a track.

% The algorithm involves two steps:
% 
% Step 1: Compute the cost of assigning every detection to each track using the distance method of the vision.KalmanFilter System object™. The cost takes into account the Euclidean distance between the predicted centroid of the track and the centroid of the detection. It also includes the confidence of the prediction, which is maintained by the Kalman filter. The results are stored in an MxN matrix, where M is the number of tracks, and N is the number of detections.
% 
% Step 2: Solve the assignment problem represented by the cost matrix using the assignDetectionsToTracks function. The function takes the cost matrix and the cost of not assigning any detections to a track.
% 
% The value for the cost of not assigning a detection to a track depends on the range of values returned by the distance method of the vision.KalmanFilter. This value must be tuned experimentally. Setting it too low increases the likelihood of creating a new track, and may result in track fragmentation. Setting it too high may result in a single track corresponding to a series of separate moving objects.
% 
% The assignDetectionsToTracks function uses the Munkres' version of the Hungarian algorithm to compute an assignment which minimizes the total cost. It returns an M x 2 matrix containing the corresponding indices of assigned tracks and detections in its two columns. It also returns the indices of tracks and detections that remained unassigned.

    function [assignments, unassignedTracks, unassignedDetections] = ...
            detectionToTrackAssignment(ot)

        ot.nTracks = length(ot.tracks);
        ot.nDetections = size(ot.centroids, 1);

        % Compute the cost of assigning each detection to each track.
        ot.cost = zeros(ot.nTracks, ot.nDetections);
        for i = 1:ot.nTracks
            ot.cost(i, :) = distance(ot.tracks(i).kalmanFilter, ot.centroids);
        end

        % Solve the assignment problem.
        ot.costOfNonAssignment = 10;
        [assignments, unassignedTracks, unassignedDetections] = ...
            assignDetectionsToTracks(ot.cost, ot.costOfNonAssignment);
        ot.assignments = assignments;
        ot.unassignedTracks = unassignedTracks;
        ot.unassignedDetections = unassignedDetections;
        assignin("base", "ot", ot)
    end

%% Update Assigned Tracks 
%The updateAssignedTracks function updates each assigned track with the corresponding detection. 
% It calls the correct method of vision.KalmanFilter to correct the location estimate.
% Next, it stores the new bounding box, and increases the age of the track and the total visible count by 1. 
% Finally, the function sets the invisible count to 0.

function updateAssignedTracks(ot)
        numAssignedTracks = size(ot.assignments, 1);

        for i = 1:numAssignedTracks
            trackIdx = ot.assignments(i, 1);
            detectionIdx = ot.assignments(i, 2);
            centroid = ot.centroids(detectionIdx, :);
            bbox = ot.bboxes(detectionIdx, :);

            % Correct the estimate of the object's location
            % using the new detection.
            correct(ot.tracks(trackIdx).kalmanFilter, centroid);

            % Replace predicted bounding box with detected
            % bounding box.
            ot.tracks(trackIdx).bbox = bbox;

            % Update track's age.
            ot.tracks(trackIdx).age = ot.tracks(trackIdx).age + 1;

            % Update visibility.
            ot.tracks(trackIdx).totalVisibleCount = ...
                ot.tracks(trackIdx).totalVisibleCount + 1;
            ot.tracks(trackIdx).consecutiveInvisibleCount = 0;
        end
        assignin("base", "ot", ot)
    end


%% Update unassigned tracks...
% Mark each unassigned track as invisible, and increase its age by 1.
    function updateUnassignedTracks(ot)
        for i = 1:length(ot.unassignedTracks)
            ind = ot.unassignedTracks(i);
            ot.tracks(ind).age = ot.tracks(ind).age + 1;
            ot.tracks(ind).consecutiveInvisibleCount = ...
                ot.tracks(ind).consecutiveInvisibleCount + 1;
        end
        assignin("base", "ot", ot)
    end


%% Delete Lost Tracks

%The deleteLostTracks function deletes tracks that have been invisible for too many consecutive frames. 
% It also deletes recently created tracks that have been invisible for too many frames overall.

function deleteLostTracks(ot, invisibleForTooLong, ageThreshold, visibilityThresh)
        if isempty(ot.tracks)
            return;
        end

        

        % Compute the fraction of the track's age for which it was visible.
        ages = [ot.tracks(:).age];
        totalVisibleCounts = [ot.tracks(:).totalVisibleCount];
        visibility = totalVisibleCounts ./ ages;

        % Find the indices of 'lost' tracks.
        lostInds = (ages < ageThreshold & visibility < visibilityThresh) | ...
            [ot.tracks(:).consecutiveInvisibleCount] >= invisibleForTooLong;

        % Delete lost tracks.
        ot.tracks = ot.tracks(~lostInds);
        assignin("base", "ot", ot)
    end

%% Create New Tracks

%Create new tracks from unassigned detections. 
% Assume that any unassigned detection is a start of a new track.
% In practice, you can use other cues to eliminate noisy detections, such as size, location, or appearance.


function createNewTracks(ot)
        centroids = ot.centroids(ot.unassignedDetections, :);
        bboxes = ot.bboxes(ot.unassignedDetections, :);

        for i = 1:size(centroids, 1)

            centroid = centroids(i,:);
            bbox = bboxes(i, :);

            % Create a Kalman filter object.
            kalmanFilter = configureKalmanFilter('ConstantVelocity', ...
                centroid, [200, 50], [100, 25], 100);

            % Create a new track.
            newTrack = struct(...
                'id', ot.nextId, ...
                'bbox', bbox, ...
                'kalmanFilter', kalmanFilter, ...
                'age', 1, ...
                'totalVisibleCount', 1, ...
                'consecutiveInvisibleCount', 0);

            % Add it to the array of tracks.
            ot.tracks(end + 1) = newTrack;

            % Increment the next id.
            ot.nextId = ot.nextId + 1;
        end
        assignin("base", "ot", ot)
end



%% Display Tracking Results

% The displayTrackingResults function draws a bounding box and label ID for each track on the video frame and the foreground mask. 
% It then displays the frame and the mask in their respective video players.

    function displayTrackingResults(ot)
        % Convert the frame and the mask to uint8 RGB.
        frame = im2uint8(ot.frame);
        mask = uint8(repmat(ot.mask, [1, 1, 3])) .* 255;

        minVisibleCount = 8;
        if ~isempty(ot.tracks)

            % Noisy detections tend to result in short-lived tracks.
            % Only display tracks that have been visible for more than
            % a minimum number of frames.
            reliableTrackInds = ...
                [ot.tracks(:).totalVisibleCount] > minVisibleCount;
            reliableTracks = ot.tracks(reliableTrackInds);

            % Display the objects. If an object has not been detected
            % in this frame, display its predicted bounding box.
            if ~isempty(reliableTracks)
                % Get bounding boxes.
                bboxes = cat(1, reliableTracks.bbox);

                % Get ids.
                ids = int32([reliableTracks(:).id]);

                % Create labels for objects indicating the ones for
                % which we display the predicted rather than the actual
                % location.
                labels = cellstr(int2str(ids'));
                predictedTrackInds = ...
                    [reliableTracks(:).consecutiveInvisibleCount] > 0;
                isPredicted = cell(size(labels));
                isPredicted(predictedTrackInds) = {' predicted'};
                labels = strcat(labels, isPredicted);

                % Draw the objects on the frame.
                frame = insertObjectAnnotation(frame, 'rectangle', ...
                    bboxes, labels);

                % Draw the objects on the mask.
                mask = insertObjectAnnotation(mask, 'rectangle', ...
                    bboxes, labels);
            end
        end

        % Display the mask and the frame.
        %obj.maskPlayer.step(mask);
        %obj.videoPlayer.step(frame);
        figure(986)
        
        %imshow(frame)
        subplot(2, 1, 1)
        imshow(frame,'InitialMagnification',2000,'Interpolation',"bilinear")
       
        subplot(2, 1, 2)
        imshow(mask,'InitialMagnification',2000,'Interpolation',"bilinear")
        
        pause(0.0001)

        assignin("base", "ot", ot)
    end


  %% End of optical tracking algorithm...



  %%




function [BW,maskedImage, centers, radii, metric] = segmentImage(X)
%segmentImage Segment image using auto-generated code from Image Segmenter app
%  [BW,MASKEDIMAGE] = segmentImage(X) segments image X using auto-generated
%  code from the Image Segmenter app. The final segmentation is returned in
%  BW, and a masked image is returned in MASKEDIMAGE.

% Auto-generated by imageSegmenter app on 01-Nov-2023
%----------------------------------------------------


% Find circles
[centers,radii,metric] = imfindcircles(X,[6 23],'ObjectPolarity','dark','Sensitivity',0.85); %% Note that senstivity here refers to this 
BW = false(size(X,1),size(X,2));
[Xgrid,Ygrid] = meshgrid(1:size(BW,2),1:size(BW,1));
for n = 1:numel(radii)
    BW = BW | (hypot(Xgrid-centers(n,1),Ygrid-centers(n,2)) <= radii(n));
end

% Create masked image.
maskedImage = X;
maskedImage(~BW) = 0;
end


%% Functions:

function [BW,maskedImage, centers, radii, metric] = segmentImageparams(X, senstivity, polarity, radmin, radmax)
%segmentImage Segment image using auto-generated code from Image Segmenter app
%  [BW,MASKEDIMAGE] = segmentImage(X) segments image X using auto-generated
%  code from the Image Segmenter app. The final segmentation is returned in
%  BW, and a masked image is returned in MASKEDIMAGE.

% Auto-generated by imageSegmenter app on 01-Nov-2023
%----------------------------------------------------
if(nargin>1)

  
else %Default arguments

    senstivity = 0.85;
    polarity = 'dark';
    radmin = 6;
    radmax = 25;

end

% Find circles
[centers,radii,metric] = imfindcircles(X,[radmin radmax],'ObjectPolarity',polarity,'Sensitivity',senstivity); %% Note that senstivity here refers to this 
BW = false(size(X,1),size(X,2));
[Xgrid,Ygrid] = meshgrid(1:size(BW,2),1:size(BW,1));
for n = 1:numel(radii)
    BW = BW | (hypot(Xgrid-centers(n,1),Ygrid-centers(n,2)) <= radii(n));
end

% Create masked image.
maskedImage = X;
maskedImage(~BW) = 0;
end

%% Find blobs instead of the circles in the image

% Find the a suitable metric that distinguishes different blobs in terms of
% not circles
% Find a image matching algorithm which distinguish


function normalizedImg = normimg(inputImg, range)
    % Check if the range is specified; if not, use the default [0, 1]
    if nargin < 2
        range = [0, 1.0];
    end

    % Perform contrast adjustment to normalize the image
    normalizedImg = imadjust(im2double(inputImg), range, []);

 
end



%% Selectively dilate and erode 

function[result_image] =  clean_erode_dilate(masktemp, threshold, disc_size)
binary_image = masktemp;
%
if nargin == 1
    % Define the threshold for blob area
    threshold = 20; % Adjust the threshold value as needed
    disc_size = 7; % Disc size for dilating 
    
end



% Label the connected components (blobs) in the binary image
labeled_image = bwlabel(binary_image, 8); % 8 connectivity

% Calculate the area of each blob
blob_areas = regionprops(labeled_image, 'Area');
blob_areas = [blob_areas.Area];

% Create a mask to identify blobs with areas larger than the threshold
large_blob_mask = blob_areas > threshold;

% Initialize the result image
result_image = zeros(size(binary_image));

% Dilate the large blobs while keeping the small blobs unchanged
for label = 1:max(labeled_image(:))
    if large_blob_mask(label)
        % Dilate the large blob using imdilate
        se = strel('disk', disc_size); % Adjust the structuring element as needed
        dilated_blob = imdilate(labeled_image == label, se);
        result_image = result_image | dilated_blob;
    else
        % Copy the small blob as is
        result_image = result_image | (labeled_image == label);
        %small_blob_mask = (labeled_image == label);
        %result_image(small_blob_mask) = 0;
    end
end

% Display or save the result
%imshow(result_image); % Display the result
%%imwrite(result_image, 'result_image.png'); % Save the result to a file
end

