To execute the experiments the workflows have to be imported in KNIME v 4.0.0 IDE.
The datasets are available in the folder "datasets".

To run the experiments with a memory of 2000 samples the following data are necessary (the way to use them is descried into the workflow annotations):

For the RRF (Reinforced Random Forest) approach
 -----------------------
| dataset  | iterations |
 -----------------------
| codec    | 1 - 613    |  
| compress | 2 - 626    |
| imaging  | 1 - 375    |
| io       | 2 - 386    |   
| lang     | 1 - 520    |
| math     | 1 - 111    |
 -----------------------

For the Static Random Forest and kNN approaches
 ---------------------------------------
| dataset  | iterations | num of samples|
 ---------------------------------------
| codec    | 362 - 613  |  2000         |
| compress | 99 - 626   |  2005         |
| imaging  | 120 - 375  |  2024         |
| io       | 150 - 386  |  2042         |
| lang     | 92 - 520   |  2033         |
| math     | 34 - 111   |  2045         |
 ----------------------------------------

To perform the sensitivity analysis the values for 1500, 1000, and 500 are the following:
For the Static Random Forest and kNN approaches

1500
 ---------------------------------------
| dataset  | iterations | num of samples|
 ---------------------------------------
| codec    | 266 - 613  |  1503         |
| compress | 83 - 626   |  1540         |
| imaging  | 90 - 375   |  1509         |
| io       | 109 - 386  |  1514         |
| lang     | 68 - 520   |  1527         |
| math     | 28 - 111   |  1500         |
 ----------------------------------------

1000
 ---------------------------------------
| dataset  | iterations | num of samples|
 ---------------------------------------
| codec    | 178 - 613  |  1001         |
| compress | 55 - 626   |  1002         |
| imaging  | 63 - 375   |  1025         |
| io       | 81 - 386   |  1004         |
| lang     | 47 - 520   |  1003         |
| math     | 19 - 111   |  1035         |
 ----------------------------------------

500
 ---------------------------------------
| dataset  | iterations | num of samples|
 ---------------------------------------
| codec    | 124 - 613  |  503          |
| compress | 25 - 626   |  511          |
| imaging  | 42 - 375   |  501          |
| io       | 51 - 386   |  502          |
| lang     | 23 - 520   |  512          |
| math     | 8 - 111    |  618          |
 ---------------------------------------

NB.: in the case of static approaches the repetitions must be performed manually