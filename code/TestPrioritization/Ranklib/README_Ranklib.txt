To run the experiments with RankLib, two scripts are available.

repetition_invocation.sh
To run the experiments for each dataset
repetitions.sh
To run the experiment 30 times for each ranker in in a certain dataset

The folder "normal" or "injected" must be changed manually in the script repetitions.sh before running the experiments.

The results will be copied in the folder of the correspondent dataset (e.g. normal/codec).
____________________________________________________________
To run the sensitivity analysis, two scripts are available.

sensitivity_invocation.sh
To run the experiments for each dataset
sensitivity.sh
To run the experiment for each ranker in in a certain dataset

The folder "normal" or "injected" must be changed manually in the script repetitions.sh before running the experiments.
Also the size of the sensitivity must be changed manually: the parameter sensitivity in sensitivity.sh must be stetted at 500, 1000, 1500 or 2000

The results will be copied in the folder of the correspondent dataset (e.g. normal/codec).
