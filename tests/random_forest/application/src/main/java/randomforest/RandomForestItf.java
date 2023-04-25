package randomforest;

import es.bsc.compss.types.annotations.Parameter;
import es.bsc.compss.types.annotations.parameter.Direction;
import es.bsc.compss.types.annotations.task.Method;

import data.dataset.DoubleDataSet;
import data.dataset.IntegerDataSet;
import data.tree.TreeFitConfig;



public interface RandomForestItf {

    @Method(declaringClass = "randomforest.RandomForest")
    IntegerDataSet randomSelection(
        @Parameter(direction = Direction.IN) int lowerBoundary,
        @Parameter(direction = Direction.IN) int upperBoundary,
        @Parameter(direction = Direction.IN) int numElements,
        @Parameter(direction = Direction.IN) long randomSeed,
        @Parameter(direction = Direction.IN) int numeroTasca
    );

    @Method(declaringClass = "data.tree.TreeTrainer")
    void trainTreeWithDataset(
        @Parameter(direction = Direction.IN) DoubleDataSet samples,
        @Parameter(direction = Direction.IN) IntegerDataSet classification,
        @Parameter(direction = Direction.IN) IntegerDataSet selection,
        @Parameter(direction = Direction.IN) TreeFitConfig config,
        @Parameter(direction = Direction.IN) long seed,
        @Parameter(direction = Direction.IN) int numeroTasca
    );
    
}