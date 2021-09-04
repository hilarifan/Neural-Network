import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.util.Scanner;
import java.io.File;
/**
 * This class creates a simple connected network with unfixed number of layers and nodes. A 3-layer network can train using back propagation.
 *
 * NeuralNet                - constructor with given # of input, output, hidden layers, and hidden layer nodes
 * maxNodes                 - finds the maximum number of nodes of all the layers
 * setWeight                - sets all of the weights to the user's input
 * randomizeWeight          - sets all of the weights to a random value between a min and max that the user can choose
 * getRandomWeight          - gives randomized double between a min and max
 * threshold                - threshold function is currently sigmoid of the parameter, the dot product
 * thresholdDeriv           - deriv of the threshold function, which is f * (1 - f)
 * evaluate                 - computes the values of each activation node using the dot product and connectivity matrix and
 *                            the values needed to train the network (lower omega, theta)
 * train                    - trains the 3-layer network using back propagation
 * error                    - computes the error from the lower omega
 * partialDeriv             - computes the partial derivative of weights for any # nodes in 3 layer network
 *
 * @author Hilari Fan
 * @version 9/6/19 (date of creation)
 */
public class NeuralNet
{
   public int inputNodes;
   public int[] hiddenLayerNodes;
   public int outputNodes;
   public double[][] activation;
   public double[][][] weight;
   public double[][] theta;
   public double[] loweromega;
   public double[] T;
   public double[][] omega;

   /**
    * constructor for NeuralNet class
    * (I sometimes comment with abbreviations like act. for activation, hl for hidden layer)
    * @param input the number of input nodes
    * @param hiddenlayer the number of hidden layers, each index is # nodes in each hidden layer
    * @param output the number of output nodes
    */
   public NeuralNet(int input, int[] hiddenlayer, int output)
   {
      inputNodes = input;

      hiddenLayerNodes = new int[hiddenlayer.length];
      for (int i = 0; i < hiddenLayerNodes.length; i++)
      {
         hiddenLayerNodes[i] = hiddenlayer[hiddenLayerNodes.length - 1 - i]; // reverse index of hl since activation layer index and weights reversed too
      }

      outputNodes = output;

      activation = new double[hiddenLayerNodes.length + 2][maxNodes()]; // # of activation layers is # hl + input + output
      weight = new double[activation.length - 1][maxNodes()][maxNodes()];

      theta = new double[activation.length - 1][maxNodes()]; // theta is calculated for each layer but the input layer

      T = new double[outputNodes];
      loweromega = new double[outputNodes];
      omega = new double[hiddenLayerNodes.length][maxNodes()]; // 2d array since there is an omegaj and omegak
   }

   /**
    * finds the maximum number of nodes in a layer (checks input, each hidden, and output layer)
    * @return the max number of nodes in a layer
    */
   public int maxNodes()
   {
      int max = 0;

      for (int i = 0; i < hiddenLayerNodes.length; i++) // finds the max nodes in the hidden layers
      {
         if (hiddenLayerNodes[i] > max)
         {
            max = hiddenLayerNodes[i];
         }
      }

      if (inputNodes > max)
      {
         max = inputNodes;
      }

      if (outputNodes > max)
      {
         max = outputNodes;
      }

      return max;
   }

   /**
    * sets all the weights to the user's input (not randomized)
    * @param scan the scanner for the file with the weight numbers
    */
   public void setWeight(Scanner scan)
   {
      int maxfrom = 0;
      int maxto = 0;

      for (int m = weight.length - 1; m >= 0; m--)
      {
         if (m == weight.length - 1)
         {
            maxfrom = inputNodes; // reversed indices so weight.length - 1 is now from input nodes
         }
         else
         {
            maxfrom = hiddenLayerNodes[m]; // otherwise, weights come from the hidden layer m since hl reversed too
            // where the layer on left is hln[1] and right is hln[0]
         }

         for (int k = 0; k < maxfrom; k++)
         {
            if (m == 0)
            {
               maxto = outputNodes; // weights going to the output nodes since 0 is now the output layer
            }
            else
            {
               maxto = hiddenLayerNodes[m - 1]; // layer m - 1 is right side since reversed indexing causes m - 1 to be the right of m
            }

            for (int j = 0; j < maxto; j++)
            {
               weight[m][k][j] = scan.nextDouble();
            }
         } // for (int k = 0; k < maxfrom; k++)
      } // for (int m = weight.length - 1; m >= 0; m--)
   }

   /**
    * randomizes all of the weights to values between a range that the user specifies
    * @param scan the scanner of the file with the range for the randomized weights
    */
   public void randomizeWeight(Scanner scan)
   {
      double min = scan.nextDouble();
      double max = scan.nextDouble();

      int maxfrom = 0;
      int maxto = 0;

      for (int m = weight.length - 1; m >= 0; m--)
      {
         if (m == weight.length - 1)
         {
            maxfrom = inputNodes; // reversed indices (same idea as setWeight)
         }
         else
         {
            maxfrom = hiddenLayerNodes[m];
         }

         for (int k = 0; k < maxfrom; k++)
         {
            if (m == 0)
            {
               maxto = outputNodes;
            }
            else
            {
               maxto = hiddenLayerNodes[m - 1];
            }

            for (int j = 0; j < maxto; j++)
            {
               weight[m][k][j] = getRandomWeight(min, max); // call helper method
            }
         } // for (int k = 0; k < maxfrom; k++)
      } // for (int m = weight.length - 1; m >= 0; m--)
   }

   /**
    * gets a random weight between the min and max parameters (helper method for randomizeWeight)
    * @param min the min of range
    * @param max the max of range
    * @return random weight
    */
   public double getRandomWeight(double min, double max)
   {
      return Math.random() * (max - min) + min;
   }

   /**
    * puts a value into the threshold function, which is currently sigmoid
    * @param num the value (from the dot products) to be put in sigmoid function
    * @return f(num) where f is the sigmoid function
    */
   public double threshold(double num)
   {
      return 1.0 / (1.0 + Math.exp(-num));
   }

   /**
    * gets the derivative of the threshold function
    * @param value the dot products or theta
    * @return the deriv of the threshold of the value
    */
   public double thresholdDeriv(double value)
   {
      return threshold(value) * (1 - threshold(value));
   }

   /**
    * generates the values of each activation using dot products and threshold function &
    * sets values for theta and loweromega that will be used for back propagation training
    * for 3 layer network
    */
   public void evaluate()
   {
      for (int i = 0; i < outputNodes; i++)
      {
         theta[0][i] = 0.0; // set to 0 since it is accumulated in the method and needs to start at 0 for each output

         for (int j = 0; j < hiddenLayerNodes[0]; j++) // reminder that hln[0] is the preceding layer of the output layer
         {
            theta[1][j] = 0.0; // set to 0 (same idea as theta[2][i])

            for (int k = 0; k < hiddenLayerNodes[1]; k++)
            {
               theta[2][k] = 0.0;

               for (int m = 0; m < inputNodes; m++)
               {
                  theta[2][k] += activation[3][m] * weight[2][m][k]; // theta 2 is for the left hidden layer, 1 for right hl, 0 for output
               }

               activation[2][k] = threshold(theta[2][k]);
               theta[1][j] += activation[2][k] * weight[1][k][j];
            } // for (int k = 0; k < hiddenLayerNodes[1]; k++)

            activation[1][j] = threshold(theta[1][j]);
            theta[0][i] += activation[1][j] * weight[0][j][i];
         } // for (int j = 0; j < hiddenLayerNodes[0]; j++)

         activation[0][i] = threshold(theta[0][i]);
         loweromega[i] = T[i] - activation[0][i];
      } // for (int i = 0; i < outputNodes; i++)
   }

   /**
    * trains the network / minimizes the error for a 3 layer network of any # of outputs or hidden layer nodes by back propagation
    * back propagation's forward calculations in evaluate and backward calculations in partialDeriv
    * (no adaptive learning or weight roll back used)
    * @param scan the scanner of the file with all the information needed to train the network
    */
   public void train(Scanner scan)
   {
      try
      {
         int testcases = scan.nextInt();

         String rOrS = scan.next();
         if (rOrS.contains("r"))
         {
            randomizeWeight(scan);
         }
         else
         {
            setWeight(scan);
         }

         double lambda = scan.nextDouble();

         double dif = scan.nextDouble(); // threshold given by user to stop training

         double maxiterations = scan.nextInt();

/**
 * I put all of my input activations for each test case in 1 array where the input activation of the first test case are
 * indices 0 to # of input activations - 1, then followed by the second test case's input activations and etc.
 * basically all input activations for first test case, then all for the next cases that follow
 */
         double[] inputactivations = new double[testcases * inputNodes];

         double[] outputactivations = new double[testcases * outputNodes]; // same idea as input activation for the output activations
         for (int num = 0; num < testcases; num++) // for each test case gets the input activations and expected output(s)
         {
            for (int m = 0; m < inputNodes; m++)
            {
               inputactivations[num * inputNodes + m] = scan.nextDouble(); // putting the value into the array in the format above
            }

            for (int i = 0; i < outputNodes; i++)
            {
               outputactivations[num * outputNodes + i] = scan.nextDouble();
            }
         }

         // create all variables

         double error = 0.0;
         double totalerror = Integer.MAX_VALUE; // set to max val to get into the while loop
         int iterations = 0;

         while (totalerror > dif && iterations < maxiterations)
         {
            totalerror = 0.0; // set to 0 since it is accumulated for each iteration and it should not carry over the val from the previous iteration
            iterations++;

            for (int num = 0; num < testcases; num++)
            {
               for (int k = 0; k < inputNodes; k++) // sets the activations to the correct input activations for the certain test case
               {
                  activation[3][k] = inputactivations[num * inputNodes + k]; // activation changed to a[3][k] to reflect reversed indices
               }

               for (int i = 0; i < outputNodes; i++) // sets the activations to the correct expected output activations for the certain test case
               {
                  T[i] = outputactivations[num * outputNodes + i];
               }

               evaluate();

               partialDeriv(lambda);

               evaluate(); // generates the activations (lower omega and theta too) for the new weights that have added the delta w

               error = error();
               totalerror += error * error; // sums the total error (square root will be placed afterwards)
            } // for (int num = 0; num < testcases; num++)

            totalerror = Math.sqrt(totalerror);
         } // while (totalerror > dif && iterations < maxiterations)


/**
 * this section is just reevaluating all the testcases using the final weight values and
 * putting all of the final values into files: total error, error for each test case, outputs for each test case,
 * expected outputs, lambda, # of iterations
 */
         File file = new File("outputs"); // I use two different files since outputs is used for the bitmap so it only contains the outputs
         FileWriter fw = new FileWriter(file.getAbsoluteFile());
         BufferedWriter bw = new BufferedWriter(fw);

         File file2 = new File("results"); // results has all the final values
         FileWriter fw2 = new FileWriter(file2.getAbsoluteFile());
         BufferedWriter bw2 = new BufferedWriter(fw2);

         totalerror = 0.0; // set to 0.0 since after training, totalerror not equal to 0 so it needs to be reset

         for (int num = 0; num < testcases; num++)
         {
            for (int k = 0; k < inputNodes; k++)
            {
               activation[3][k] = inputactivations[num * inputNodes + k]; // same idea as above
            }

            for (int i = 0; i < outputNodes; i++)
            {
               T[i] = outputactivations[num * outputNodes + i];
            }

            bw2.write("Test Case " + (num + 1) + "\n");

            evaluate();

            for (int i = 0; i < outputNodes; i++)
            {
               bw2.write("Final Output: " + activation[0][i] + "\n");
               bw.write(activation[0][i] + "\n");

               bw2.write("Expected Output: " + T[i] + "\n");
            }

            error = error();

            bw2.write("Final Error: " + error + "\n" + "\n");
            totalerror += error * error; // I add all the error^2 so that at the end I will do a sqrt
         } // for (int num = 0; num < testcases; num++)

         bw2.write("\n" + "lambda: " + lambda + "\n");
         bw2.write("Total Error: " + Math.sqrt(totalerror) + "\n");
         bw2.write("# of Iterations: " + iterations + "\n");

         bw.close();
         bw2.close();
      }
      catch (Exception e)
      {
         System.out.println(e);
         e.printStackTrace();
      }
   }

   /**
    * gets the total error for a single training set model
    * @return the error
    */
   public double error()
   {
      double error = 0.0;
      for (int i = 0; i < outputNodes; i++)
      {
         error += (loweromega[i]) * (loweromega[i]);
      }
      return 0.5 * error;
   }

   /**
    * finds the partial derivative of the weights and adds it to the existing weights with respect to the error by back propagation
    * for 3 layer network
    * @param lambda the lambda from the user
    */
   public void partialDeriv(double lambda)
   {
      double psi = 0.0; // psi is used to represent psii, psij, and psik

      for (int j = 0; j < hiddenLayerNodes[0]; j++) // for the last hidden layer to output layer
      {
         omega[0][j] = 0.0; // set to 0 since do not want to add to previous iteration's omega

         for (int i = 0; i < outputNodes; i++)
         {
            psi = loweromega[i] * thresholdDeriv(theta[0][i]);
            omega[0][j] += psi * weight[0][j][i]; // accumulate omega in array since it is used for delta weight of hidden to hidden layer
            weight[0][j][i] += lambda * activation[1][j] * psi;
         }
      } // for (int j = 0; j < hiddenLayerNodes[0]; j++)


      for (int k = 0; k < hiddenLayerNodes[1]; k++)
      {
         omega[1][k] = 0.0; // same idea as above

         for (int j = 0; j < hiddenLayerNodes[0]; j++)
         {
            psi = omega[0][j] * thresholdDeriv(theta[1][j]);
            omega[1][k] += psi * weight[1][k][j]; // same accumulation for input to hidden
            weight[1][k][j] += lambda * activation[2][k] * psi;
         }
      } // for (int k = 0; k < hiddenLayerNodes[1]; k++)


      for (int m = 0; m < inputNodes; m++)
      {
         for (int k = 0; k < hiddenLayerNodes[1]; k++)
         {
            psi = omega[1][k] * thresholdDeriv(theta[2][k]);
            weight[2][m][k] += lambda * activation[3][m] * psi;
         }
      }
   }

   /**
    * tests the minimization of the error using back propagation
    * @param args argument
    */
   public static void main(String[] args)
   {
      File file = new File("/Users/hilarifan/IdeaProjects/xor/activations"); // file with all the parameters
      try
      {
         Scanner scan = new Scanner(file);
         int input = scan.nextInt();

         int[] testerHiddenLayer = new int[scan.nextInt()];
         for (int i = 0; i < testerHiddenLayer.length; i++)
         {
            testerHiddenLayer[i] = scan.nextInt();
         }

         int output = scan.nextInt();

         long startTime = System.currentTimeMillis();

         NeuralNet tester = new NeuralNet(input, testerHiddenLayer, output);

         tester.train(scan);

         System.out.print("time taken in milliseconds = ");
         System.out.println(System.currentTimeMillis() - startTime);
      }
      catch (FileNotFoundException e)
      {
         e.printStackTrace();
      }
   }
}
