# Higgs_boson_classification

File organization:
```
ml-project-1-hybrid_ml/
    | scripts
        | implementations.py
        | run.py
        | helpers.py
        | proj1_helpers.py
        | costs.py
        | gradients.py
    | Report.pdf     
```

Contents : 
- implementations.py : contains all required functions asked in the project description
- proj1_helpers.py : contains code given by the teachers (to load, predict, and generate submissions)
- helpers.py, costs.py, gradients.py: contain all the helper functions that are not in implementations.py that were used during the realization of this project
- run.py : contains the code to run that replicates submission results

Instructions to run "run.py":
- Make sure to indicate in the code the path to the files "train.csv" and "test.csv". You can find the data in https://www.aicrowd.com/challenges/epfl-machine-learning-higgs/leaderboards where you can download it. The files should be unizpped. By default, we consider they're on the data folder.
- All the data pre-processing is done within the code, you only need to run it to produce the prediction.
- After running the code, a new file called "prediction.csv" should appear in the folder, it contains the predictions for the test data.

PS: the data for this project isn't public, check the AICrowd Higgs Boson challenge for the data when it's available.
