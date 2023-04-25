# Euro-Par23 - Test - GridSearch

The first test evaluates the performance of GridSearch with cross-validation: an algorithm that exhaustively looks for all the different combinations of hyper-parameters for a particular estimator. With cross-validation, it trains and evaluates several estimators for each combination (splits), and the final score obtained for a combination is the average of the scoring of all its corresponding splits. GridSearch is one of the algorithms offered within dislib, a Python library built on top of PyCOMPSs providing distributed mathematical and machine learning algorithms. Since the original version of COMPSs does not support nested tasks, the implementation of GridSearch currently provided within dislib  delegates the detection of the tasks to the implementation of the estimator and invokes them sequentially expecting them to create all the finer-grain tasks at a time.

The conducted test finds the optimal solution among 25 or 50 combinations of values for the Gamma (5 values from 0.1 to 0.5 or 10 values from 0.1 to 1.0) and C (5 values from 0.1 to 0.5) hyper-parameters to train a Cascade-SVM classification model (CSVM). CSVM is an iterative algorithm that checks the convergence of the model at the end of every iteration; hence, its dislib implementation stops generating tasks at the end of each iteration until it evaluates the convergence of the model. This affects the parallelism of GridSearch; it does not detect tasks from a CSVM until the previous one converges. To overcome this shortcoming, the Nested version of the GridSearch algorithm encapsulates the fitting and evaluation of each estimator within a coarse-grain task. In turn, each estimator generates its corresponding finer-grain tasks achieving higher degrees of parallelism. 

# APPLICATION SOURCE CODE
Folder `application` contains the source code and datasets required to run the application. 
The `dataset` folder contains the datasets being used in the executions

# Test Execution
To launch the application, users can directly call the `launch` script within the test folder passing in the indicated parameters for the version which are:
- NUM_COMBINATIONS: number of combinations to try <25|50>
- NUM_MODELS: number of models being trained
- DATASET: Name of the dataset used for creating the model. This can be `IRIS` (default value), `DIGITS` or `AT`.

```bash
> launch <FLAT|NESTED> <25|50> [NUM_MODELS [DATASET<IRIS|DIGITS|AT>]]
```

Otherwise, the user can use the `launch_test` script in the root folder of the repository or the container passing in `gridsearch` as the first parameter. Both ways end up calling the launch script removing the application name.
```bash
> launch_test gridsearch <FLAT|NESTED> <25|50> [NUM_MODELS [DATASET<IRIS|DIGITS|AT>]]
> docker run --rm francesclordan/europar23:latest gridsearch <FLAT|NESTED> <25|50> [NUM_MODELS [DATASET<IRIS|DIGITS|AT>]]
```

The script `enqueue` can be used to submit the execution onto the queue system of a supercomputer. Execution times can be retrieved with the `get_times` script which returns a list with an entry for each execution detailing the execution id, the number of nodes used, the number of estimators and the training time in ms for each execution.
```bash
> enqueue <num_nodes> <FLAT|NESTED> <25|50> [NUM_MODELS [DATASET]]
> get_times
```
