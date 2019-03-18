# image_classification
A computer Vision algorithm which predict weather the given Image is of a car or it is of a bike.
I have used Global feature extraction and then combined them to form a single global vector.
Finally trained those extracted vector on RandomForrestClassifier to predict the output.

## Getting Started

Download the files in a folder. The folder should have Dataset Folder, an Output folder and three python codes.(from repository)
Dataset folder will contain two subfolders train and test. Train folder will have two subfolders in it namely car and motorbike containing corresponding images.

Run feature_train.py, which will extract the features of training data and will vectorise the same. Also the vectorized file will get saved in Output folder as data.h5 and labels.h5
now run feature_test.py, which will extract the features of test data and will vectorize it. Also the vectorized file will get saved in Output folder in as test_data.h5

Now to train and get output run main.py, which will train the data and it will dump the output in output1.xlxs file



