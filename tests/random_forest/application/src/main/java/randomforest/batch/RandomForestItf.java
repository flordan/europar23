package randomforest.batch;

import es.bsc.compss.types.annotations.Constraints;
import es.bsc.compss.types.annotations.Parameter;
import es.bsc.compss.types.annotations.parameter.Direction;
import es.bsc.compss.types.annotations.task.Method;

import data.dataset.DoubleDataSet;
import data.dataset.IntegerDataSet;
import data.tree.TreeFitConfig;
import randomforest.config.FitConfig;


public interface RandomForestItf {

    @Method(declaringClass = "randomforest.batch.RandomForest")
    @Constraints(isLocal=true)
    IntegerDataSet randomSelection(
        @Parameter(direction = Direction.IN) int lowerBoundary,
        @Parameter(direction = Direction.IN) int upperBoundary,
        @Parameter(direction = Direction.IN) int numElements,
        @Parameter(direction = Direction.IN) long randomSeed,
        @Parameter(direction = Direction.IN) int numeroTasca
    );

    @Method(declaringClass = "data.tree.TreeTrainer")
    @Constraints(isLocal=true)
    void trainTreeWithDataset(
        @Parameter(direction = Direction.IN) DoubleDataSet samples,
        @Parameter(direction = Direction.IN) IntegerDataSet classification,
        @Parameter(direction = Direction.IN) IntegerDataSet selection,
        @Parameter(direction = Direction.IN) TreeFitConfig config,
        @Parameter(direction = Direction.IN) long seed,
        @Parameter(direction = Direction.IN) int numeroTasca
    );

    @Method(declaringClass = "randomforest.batch.RandomForest")
    @Constraints(computingUnits = "48")
    void generateTaskBatch(
        @Parameter(direction = Direction.IN) long batchSeed,
        @Parameter(direction = Direction.IN) TreeFitConfig treeFitConfig,
        @Parameter(direction = Direction.IN) int numSamples,
        @Parameter(direction = Direction.IN) DoubleDataSet samples,
        @Parameter(direction = Direction.IN) IntegerDataSet classification,
        @Parameter(direction = Direction.IN) FitConfig config,
        @Parameter(direction = Direction.IN) int batchSize,
        @Parameter(direction = Direction.IN) int numeroTasca
    );

}
