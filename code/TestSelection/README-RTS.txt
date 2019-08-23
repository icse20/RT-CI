To run test selection, the following scripts should be used as follows: 

- push.sh is the main script. It creates a local branch from the githhub repository, with “n” commits. 

- the script then pushes the commits one by one , and invokes “script.sh”

- the latter script updates the dependencies creates a dependency file, containing all the classes depending on the modified ones: it invokes “dependency_searching.py” which queries the "Understand" database of the application to look for dependencies. 
NOTE: Before using the scripts, you need UNDERSTAND tool (https://scitools.com) to extract metrics from the code. For a given subject, you need to create the Understand database after the first “push" that you do, and save it (the paths in the scripts need to be updated accordingly). Each invokation of “script.sh” updates the database (hence, when the application changes after a commit, the database is updated accordingly and dependenices are collected on the updated database). 

- then, "testing.sh dependsby.txt” tests the selected classes

- Finally, “ data_collection_new.py” collects the data. Each row of the output is a test class, each column  is a “metric” (either complexity metrics extracted from understand and the outcome of the test cases)

