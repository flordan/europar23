package simple;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;


public class SimpleGOS {

    /**
     * The entry point of application.
     */
    public static void main(String[] args) {
        // Check and get parameters
        if (args.length != 1) {
            System.out.println("[ERROR] Bad number of parameters");
            System.out.println(" Usage: simple.Simple <counterValue>");
            System.exit(-1);
        }

        String counterName = "counter";

        int initialValue = Integer.parseInt(args[0]);
        // ------------------------------------------------------------------------ // Write value
        try {
            FileOutputStream fos = new FileOutputStream(counterName);
            fos.write(initialValue);
            fos.close();
        } catch (IOException ioe) {
            ioe.printStackTrace();
            System.exit(-1);
        }

        // ------------------------------------------------------------------------ // Execute increment
        SimpleGOSImpl.taskCPU(counterName);
        SimpleGOSImpl.taskGPU(counterName);
        SimpleGOSImpl.taskSoftware(counterName);

        // ------------------------------------------------------------------------ // Read new value
        try {
            FileInputStream fis = new FileInputStream(counterName);
            System.out.println("Final counter value is " + fis.read());
            fis.close();
        } catch (IOException ioe) {
            ioe.printStackTrace();
            System.exit(-1);
        }
    }

    /*
     * public static void main(String[] args) { for (int i = 0; i < 3; i++) { SimpleGOSImpl.increment(); } }
     */
}
