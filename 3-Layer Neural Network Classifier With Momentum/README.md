# Human Activity Recognition Classifier with and without Momentum

  In this project, we are classifying human activities using the sensory data obtained from a
Samsung Galaxy S2. The classes are: 1 WALKING, 2 WALKING_UPSTAIRS, 3 WALKING_DOWNSTAIRS, 4
SITTING, 5 STANDING, 6 LAYING. The sensory data is collected at 50Hz. Using these parameters, some
statistical values obtained from the feature are added as a feature to the dataset. Some of these
features are: mean, standard deviation, min, max. These statistical features enables us to get
information about the sensory data obtained before the time of prediction.

  For this project, I implemented a 3-layer neural network. The gradient of the error function with a specific
parameter is calculated using the chain rule. We also add a moment term where we add the previous
gradient multiplied with a momentum coefficient to the calculation of the current gradient.
