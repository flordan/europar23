package checkpointPolicyEvery3tasks;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.FileNotFoundException;


public class SimpleImpl {

    public static void increment(String counterFile) {
        // Read value
        try {
            FileInputStream fis = new FileInputStream(counterFile);
            int count = fis.read();
            fis.close();

            // Write new value
            FileOutputStream fos = new FileOutputStream(counterFile);
            fos.write(++count);
            fos.close();
        } catch (FileNotFoundException fnfe) {
            fnfe.printStackTrace();
            System.exit(-1);
        } catch (IOException ioe) {
            ioe.printStackTrace();
            System.exit(-1);
        }
    }
}
