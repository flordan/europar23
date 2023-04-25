package data.tree;

import data.dataset.DoubleDataSet;
import data.dataset.IntegerDataSet;


public class TreeTrainer {
    
    public static void trainTreeWithDataset(DoubleDataSet samples, IntegerDataSet classification,
        IntegerDataSet selection, TreeFitConfig config, long seed, int numeroTasca) {
        System.out.println("______ tasca trainTreeWithDataset " + Integer.toString(numeroTasca));
        Tree.trainTreeWithDataset(samples, classification, selection, config, seed);
    }
}
