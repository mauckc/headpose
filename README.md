## About headpose
Process camera feed for head pose estimation is a Python application for computer vision live face parameterization

Using dlib's face landmark predictor, I added my implementation of a real-time by building a graphics pipeline to support the 2D to 3D head pose estimation method built by Satya Mallick in the referenced code below.

---
### Referenced Code
* https://www.learnopencv.com/head-pose-estimation-using-opencv-and-dlib
* https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python

### Dependencies
You need to have Python 2.6+ as a minimum and:

* [NumPy](http://numpy.scipy.org/)
* [OpenCV 3](http://opencv.org/) Prefer OpenCV 3.4+
* [Dlib](http://dlib.net/)
* [Imutils](https://github.com/jrosebr1/imutils)

<p align="center">
  <img src="https://github.com/mauckc/headpose/blob/master/media/elon-sample.gif"/>
</p>

### Structure

*python/*  the code.

*models/*  contains the models used in this example we use Facial Landmark detection 68 points.
           *one must download shape_detector_68_facial_landmarks.dat because it is too large a file to host here.

*media/*  contains images and video. 

### Installation

Open a terminal in the headpose directory and run (with sudo if needed on your system):

	pip install -r requirements.txt

Now you should have installed the necessary packages

You still need to download dlib model file: "shape_predictor_68_face_landmarks.dat"

This is a location where that file is hosted: https://github.com/AKSHAYUBHAT/TensorFace/blob/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat

Put this model file "shape_predictor_68_face_landmarks.dat" located in YOUR_MODEL_DOWNLOAD_PATH into the model folder headpose/models

 	cp YOUR_MODEL_DOWNLOAD_PATH/shape_predictor_68_face_landmarks.dat headpose/models/shape_predictor_68_face_landmarks.dat
	
Give privilages to run the shell script to start application

	chmod +x run.sh

Then run the shell script

	./run.sh
	
in your Python session or script. Try one of the sample code examples to check that the installation works.

For good resources on these topics see:

Detecting face landmarks: Adrian Rosebrock implementation of dlib's shape_predictor's for face landmark detection
*[Face Landmark Detection](https://www.learnopencv.com/head-pose-estimation-using-opencv-and-dlib/)

Pose estimation: Satya Mallick's implementation of OpenCV's PnP function
*[Pose Estimation](https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/) 

Using this and this 
to get this information

and this to take that information which would produce this output.

I wanted to write a graphics pipeline that would produce this in th
I did this

<p align="center">
  <img src="https://github.com/mauckc/headpose/blob/master/media/obama-sample.gif"/>
</p>

### License

All code in this project is provided as open source under the MIT license


---
-Ross Mauck


