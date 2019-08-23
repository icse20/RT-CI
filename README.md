# Learning-to-Rank vs Ranking-to-Learn: Strategies for Regression Testing in Continuous Integration

This repository is a companion page for a research paper submitted to the 42nd International Conference on Software Engineering ([ICSE'20](https://conf.researchr.org/home/icse-2020)). It contains all the material required for replicating our experiments, including: the implementation of the algorithms (along with supplementary tools), the input data, and the obtained results. Some additional results, not included in the paper for the sake of space, are also provided ("SupplementalMaterial.pdf").

Experiment Replication
---------------
In order to replicate the experiment follow these steps:

### Getting started

1. Clone the repository 
   - `git clone https://github.com/icse20/RT-CI/`
   
2. Step 2. Run test selection first. This will produce datasets (one per subject) with rows representing the selected test classes for each commit and columns being the metrics associated with that test class (i.e., the code metrics of the class(es) under test and of the test cases within the test class). To run test selection, please refer to the readme files within the code/TestSelection folder. 

3. Step 3. Run test prioritization. To this am, select the type of algorithm to run. Depending on this, the KNIME tool (for KNN, RF and RL-RF algorithms), the RankLib library (for LTR algorithms), or our Python code (for RL and RL-MLP) need to be used, which provide implementations for the mentioned algorithms that we have experimented. To run test prioritization, please refer to the readme files within the code/TestPrioritization folder. It also contains the input datasets in the format required by the tools and support scripts to do the same analysis that are in the paper. 

Directory Structure
---------------
This is the root directory of the repository. The directory is structured as follows:

    RT-CI
     .
     |
     |--- code/          Implementation of the algorithms and scripts to execute the experiments.
     |
     |--- datasets/      The datasets used for running the experiments.
     |
     |--- results/       Experiment results and related data, supplemental material complementing the paper's result.
     |
     |--- tools/         Links to additional tools used in our experiments.
  
