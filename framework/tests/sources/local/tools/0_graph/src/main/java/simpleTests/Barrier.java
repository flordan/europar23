package simpleTests;

import es.bsc.compss.api.COMPSs;
import utils.GenericObject;
import utils.TasksImplementation;


public class Barrier {

    private static final int N = 3;


    public static void main(String[] args) {
        GenericObject[] values = new GenericObject[N];

        for (int i = 0; i < N; ++i) {
            values[i] = TasksImplementation.initialize();
        }

        COMPSs.barrier();

        for (int i = 0; i < N; ++i) {
            TasksImplementation.increment(values[i]);
        }

        // Final sync
        for (int i = 0; i < N; ++i) {
            System.out.println("Value " + i + " is " + values[i].getValue());
            try {
                Thread.sleep(5_000);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }

}
