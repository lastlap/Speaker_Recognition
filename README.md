# Speaker Recognition

Identify multiple speakers using machine learning models like SVM, Decision Trees, XgBoost etc.
Mel Coefficients(MFCC) of the audio are used as features for the models

## Requirements

System packages:

	ffmpeg == 4.2.4
	python == 3.8.5
	
Python packages:

	librosa == 0.7.2
	pandas == 1.1.3
	numpy == 1.19.2
	sklearn == 0.23.2
	six == 1.15.0
	pydotplus
	xgboost
  
  
  
	
## Project Structure

Audio clips of each class are to be placed in a separate folder and their names start from '0','1','2'... with as many folders as there are speakers/classes.
Here I had used audio clips of 5 seconds each. I had 50 audio clips for each class '0','1' and '2'.


`preprocess.py` - Run this to find audio features for each clip in the dataset. It creates a .csv file.

`train.py` - Multiple models are used to predict classes. Accuracies of each model is printed out.

`test.py` - Use this to test on new audio samples.
