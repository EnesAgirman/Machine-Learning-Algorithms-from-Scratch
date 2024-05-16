# Regression with a 4-Layer Feed-Forward Neural Network

Note: In this project, I used pytorch just to do matrix multiplications on the gpu. I didn't use any of the machine learning functionality of pytorch.

  In this project, we are tackling 2 regression problems with the same neural network. The neural network is a 4-layer feed-forward neural network with dropout, weight decay, momentum and cosign annealing learning rate. We are implementing our neural network with 2 datasets:
## 1) Student Performance Dataset :
  ### Predictor Variables
  - Hours Studied: The total time of daily study in hours (1 - 9 integer)
  - Previous Scores: Scores from previous tests out of 100 (44 to 99 integer)
  - Extracurricular Activities: Participation in extracurricular activities (Yes or No).
  - Sleep Hours: The average daily sleep in hours (4 - 9 integer)
  - Sample Question Papers Practiced: The total number of practiced sample question papers (0 - 9 integer)
  ### Target Variable
  - Performance Index: Represents the overall academic success of the student (10 - 100 integer)
## 2) Graduate Admission Dataset [2]:
  ### Predictor Variables
  - GRE Score: GRE score of the student. (0 - 340 integer)
  - TOEFL Score: TOEFL score of the student. (0 - 120 integer)
  - University Rating: The rating of the university students graduated from (0 - 5 integer)
  - Statement of Purpose: The strength of the SOP (0 to 5 integer)
  - Letter of Recommendation Strength: The strength of the LOR (0 - 5 integer)
  - Undergraduate GPA: Undergraduate GPA of the student (0 - 10 integer)
  - Research Experience: If the student has research experience or not (either 1 or 0)
 ### Target Variable
  - Chance of Admit: The student’s chance of admittance to the graduate school (0 - 1 float)

These datasets are put through PCA to have 4 predictor variables in the student performance dataset and 5 predictor variables in the graduate admission dataset. The result of the PCA part is in datasets\graduate_admission/Admission_6pc.csv and datasets\student_performance/Performance.csv for the graduate admission and student performance datasets respectively.

While using these methods, we face the challenge of hyper-parameter optimisation where there are simply too many hyper-parameters for us to be able to optimize together. To be able to choose the hyper-parameters, I utilized the random search method. This method chooses random combinations of hyper-parameters and trains the algorithm. It compares the validation results and chooses the best performing set of hyper-parameters from the validation results. We set the exploration length as 50. The options for the random search was as follows:
  - max learning rate choices = [0.05, 0.01, 0.001, 0.0001] 
  - dropout 1 choices = [0, 0.1, 0.3, 0.5] 
  - dropout 2 choices = [0, 0.1, 0.3, 0.5] 
  - dropout 3 choices = [0, 0.1, 0.3, 0.5]
  - weight decay choices = [0.001, 0.01, 0.1] 
  - momentum choices = [0.2, 0.5, 0.9] 
  - epochs choices = [20, 50, 100, 200]
  - batch size choices = [10, 20, 50] 
  - hidden size 1 choices = [50, 100, 200] 
  - hidden size 2 choices = [20, 50, 100]
  - hidden size 3 choices = [10, 20, 50]

For validation, I used k-fold cross validation again, the same method used in linear and ridge regres-
sion. For the graduate admission dataset, I used 10-fold cross validation and for the student performance
dataset, we used 5-fold cross validation. I used 5-fold in student performance set instead of 10-fold
because the computing time exceeds 7-hours even when utilizing the GPU for matrix multiplications.

## Results

I implemented the algorithm described above to get the following results:
### Graduate Admission Dataset(trained on training+validation data): 
- MAE: 0.0073 
- MSE: 0.0015 
- R2: 0.0059

### Graduate Admission Optimal Hyper-parameters:
- lr max: 0.05,
- dropout 1: 0.0, 
- dropout 2: 0.3, 
- dropout3: 0.5,
- weight decay: 0.1,
- momentum: 0.2,
- epochs: 200,
- batch size: 20,
- hidden size 1: 50,
- hidden size 2: 100,
- hidden size 3: 20

## Student Performance Dataset(trained on training+validation data):
- MAE: 2.2197
- MSE: 7.5149
- R2: 1.7481
### Student Performance Optimal Hyper-parameters: 
- lr max: 0.05,
-  dropout 1: 0,
-  dropout 2: 0,
-  dropout 3: 0.1,
-  weight decay: 0.01,
-  momentum: 0.9,
-  epochs: 200,
-  batch size: 20,
-  hidden size 1: 100,
-  hidden size 2: 50,
-  hidden size 3: 10



