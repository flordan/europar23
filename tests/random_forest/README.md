# Euro-Par23 - Test - Random Forest
The second experiment consists in training a classification model using the RandomForest algorithm, which constructs a set of individual decision-trees, also known as estimators, each classifying a given input into classes based on decisions taken in random order. The final classification of the model is the aggregate of the classification of all the estimators; thus, the accuracy of the model depends on the number of estimators composing it. The training of the estimators are independent of each other, each consisting of two tasks: a first one that selects a combination of 30,000 random samples from the training set and a second one that builds the decision tree. The test uses two versions of the algorithm: one where the main task directly generates the task to train all the estimators -- Flat -- and another one where the main task generates intermediate tasks grouping the training of several estimators -- Nested. In the conducted tests, each batch trains at least 48 estimators, and if the number of estimators allows it, the number of intermediate tasks matches the number of deployed Agents.

# APPLICATION SOURCE CODE
Folder `application` contains the maven project to build the application. The source code of the application can be found in the `src/java/main` subfolder. 

The `randomforest.RandomForest` class is the main class of the `FLAT` version of the application. In this version of the algorithm, two methods are selected as tasks: `randomSelection` and `trainTreeWithDataset` as indicated in `randomforest.RandomForestItf` interface. This version accepts two parameters: the number of models to train on one execution and the number of estimators for each model. By default, it creates one model of 1 estimator.

The `randomforest.batch.RandomForest` class is the main class of the `NESTED` version of the application. In this version of the algorithm, three methods are selected as tasks: `generateTaskBatch`, `randomSelection` and `trainTreeWithDataset` as indicated in `randomforest.batch.RandomForestItf` interface. While `generateTaskBatch` can be executed by any node of the infrastructure reserving 48 cores of the node, the other two tasks can only be executed in the same node that detected them sugin a single core. This version accepts three parameters: the number of models to train on one execution, the number of estimators for each model and the size of the batch - at least, 48 . By default, it creates 1 model of 1 estimator in one single batch.

# Test Execution
To launch the application, users can directly call the `launch` script within the test folder passing in the indicated parameters for the version which are:
- NUM_MODELS: number of models being trained
- NUM_ESTIMATORS: number of estimators for each model
- BATCH_SIZE: number of estimators in computed by each batch (only affects to the NESTED version) 

```bash
> launch <FLAT|NESTED> [NUM_MODELS [NUM_ESTIMATORS [BATCH_SIZE]]]
```

Otherwise, the user can use the `launch_test` script in the root folder of the repository or the container passing in `random_forest` as the first parameter. Both ways end up calling the launch script removing the application name.
```bash
> launch_test random_forest <FLAT|NESTED> [NUM_MODELS [NUM_ESTIMATORS [BATCH_SIZE]]]
> docker run --rm francesclordan/europar23:latest random_forest <FLAT|NESTED> [NUM_MODELS [NUM_ESTIMATORS [BATCH_SIZE]]]
```

The script `enqueue` can be used to submit the execution onto the queue system of a supercomputer. Execution times can be retrieved with the `get_times` script which returns a list with an entry for each execution detailing the execution id, the number of nodes used, the number of estimators and the training time in ms for each execution.
```bash
> enqueue <num_nodes> <FLAT|NESTED> [NUM_MODELS [NUM_ESTIMATORS]]
> get_times
```
