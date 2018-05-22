## About headpose
Process camera feed for head pose estimation is a Python application for computer vision live face parameterization based on Adrian Rosebrock and Satya Mallick

### Dependencies
You need to have Python 2.6+ and as a minimum:

* [NumPy](http://numpy.scipy.org/)
* [OpenCV3](http://opencv.org/)

Some parts use:

* [Imutils](https://github.com/jrosebr1/imutils)

### Structure

*python/*  the code.

*models/*  ontains the models used in this example we use Facial Landmark detection 68 points.
           *one must download shape_detector_68_facial_landmarks.dat because it is too large a file to host here.

*media/*  contains images and video. 

### Installation

Open a terminal in the headpose directory and run (with sudo if needed on your system):

	pip install -r requirements.txt

Now you should have installed the necessary packages

Give privelages to run the shell script to start application

	chmod +x run.sh

Then run the shell script

        ./run.sh
	
in your Python session or script. Try one of the sample code examples to check that the installation works.

### License

All code in this project is provided as open source under the MIT license


---
-Ross Mauck
