1. the libraries required to run the project including the full version of each library

matplotlib==3.5.0
numpy==1.21.3
opencv_python==4.5.3.56
pandas==1.3.4
scikit_image==0.19.1
scikit_learn==1.0.2
skimage==0.0
tensorflow==2.7.0

2. To run the whole project you need to run in order:

Any path configuration is done in the params.
First run the scripts for generating the images:
Utils/GeneratePositiveExamples.py
Utils/GenerateNegativeExamples.py

Then run task 1 first and after that run task 2

Task 1: 
script: RunTask1.py
You can change params.model_used to swap between models used to train and predict
Just run task1 and it will generate the answers in the file specificed in params.

Task 2: RunTask2.py
Just run the script and it will generate all files
