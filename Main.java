package machinelearning;

import java.io.*;
import java.util.*;
import java.util.stream.Collectors;


public class Main {
    private List<double[]> trainingData;
    private List<Integer> trainingLabels;
    private List<double[]> testData;
    private List<Integer> testLabels;
    private int featureCount;

    public Main() {
        trainingData = new ArrayList<>();
        trainingLabels = new ArrayList<>();
        testData = new ArrayList<>();
        testLabels = new ArrayList<>();
    }
    
   // Loading data from the csv File and slpit in feathure and lablels

    public void loadData(String filename, boolean isTestData) {
        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                double[] features = new double[values.length - 1];
                for (int i = 0; i < features.length; i++) {
                    features[i] = Double.parseDouble(values[i]);
                }
                if (isTestData) {
                    testData.add(features);
                    testLabels.add(Integer.parseInt(values[values.length - 1]));
                } else {
                    trainingData.add(features);
                    trainingLabels.add(Integer.parseInt(values[values.length - 1]));
                }}
            
            
        } catch (IOException e) {
            e.printStackTrace();
   }
    }
    
    // Standardize the date by adjusting the mean and standard devaiation for each feature 

    private void standardizeData(List<double[]> data, double[] means, double[] stdDevs) {
        if (data.isEmpty() || data.get(0).length == 0) return;

        featureCount = data.get(0).length;

        if (means.length != featureCount) {
            means = new double[featureCount];
            
        }
        if (stdDevs.length != featureCount) {
        	
            stdDevs = new double[featureCount];
        }

        Arrays.fill(means, 0);
        Arrays.fill(stdDevs, 0);

        // Calculate means for each feature
        for (int i = 0; i < featureCount; i++) {
            for (double[] row : data) {
                if (row.length != featureCount) {
                	
                    throw new IllegalArgumentException("Row " + Arrays.toString(row) + " has an incorrect number of features.");
                }
                means[i] += row[i];
                
            }
            means[i] /= data.size();
        }

        // Calculate standard deviations for each feature
        
        
        for (int i = 0; i < featureCount; i++) {
            for (double[] row : data) {
                stdDevs[i] += Math.pow(row[i] - means[i], 2);
            }
            stdDevs[i] = Math.sqrt(stdDevs[i] / data.size());
        }

        // Standardize data
        for (double[] row : data) {
            for (int i = 0; i < featureCount; i++) {
                if (stdDevs[i] == 0) {
                	
                	// If std is 0, set the value 0 (avoid  division by zero)
                    row[i] = 0; 
                    
                    
                } else {
                	
                	
                    row[i] = (row[i] - means[i]) / stdDevs[i];
                }
                
                
           
            }
        }}

    private void encodeLabels() {
    	
        trainingLabels = trainingLabels.stream()
            .map(label -> label == 0 ? -1 : label)
            .collect(Collectors.toList());
    }

    private void balanceClasses() {
        List<double[]> class1Data = new ArrayList<>();
        List<double[]> class2Data = new ArrayList<>();
        List<Integer> class1Labels = new ArrayList<>();
        List<Integer> class2Labels = new ArrayList<>();
        for (int i = 0; i < trainingLabels.size(); i++) {
            if (trainingLabels.get(i) == 1) {
                class1Data.add(trainingData.get(i));
                class1Labels.add(1);
            } else {
                class2Data.add(trainingData.get(i));
                class2Labels.add(-1);
            }
        }
        while (class1Data.size() < class2Data.size()) {
            class1Data.addAll(class1Data);
            class1Labels.addAll(class1Labels);
        }
        while (class2Data.size() < class1Data.size()) {
            class2Data.addAll(class2Data);
            class2Labels.addAll(class2Labels);
        }
        trainingData = new ArrayList<>(class1Data);
        trainingData.addAll(class2Data);
        trainingLabels = new ArrayList<>(class1Labels);
        trainingLabels.addAll(class2Labels);
    }

    private double[] createPolynomialFeatures(double[] features, int degree) {
        List<Double> newFeatures = new ArrayList<>();
        for (int i = 0; i < features.length; i++) {
            for (int d = 1; d <= degree; d++) {
                newFeatures.add(Math.pow(features[i], d));
            }
        }
        return newFeatures.stream().mapToDouble(Double::doubleValue).toArray();
    }

    private void applyPolynomialFeatures(int degree) {
        List<double[]> transformedData = new ArrayList<>();
        for (double[] features : trainingData) {
            transformedData.add(createPolynomialFeatures(features, degree));
        }
        trainingData = transformedData;
        featureCount = trainingData.get(0).length;
    }
    
    // Train linear SVM using SGD
    
    

    public double[] trainLinearSVM(double C, double learningRate, int maxIterations) {
    	
        int n = trainingData.size();
        int m = featureCount;
        double[] w = new double[m];
        
        double b = 0;
        boolean converged = false;
        
        // Iterate up to maxIterations
        for (int iter = 0; iter < maxIterations && !converged; iter++) {
        	
            double prevB = b;
            for (int i = 0; i < n; i++) {
                double[] x = trainingData.get(i);
                
                int y = trainingLabels.get(i);
                double margin = y * (dotProduct(w, x) + b);
                
                if (margin < 1) {
                    for (int j = 0; j < m; j++) {
                        w[j] -= learningRate * (C * y * x[j] - 2 * w[j]);
                   }
                    b -= learningRate * (-C * y);
                } else {
                    for (int j = 0; j < m; j++) {
                        w[j] -= learningRate * (-2 * w[j]);
                    }}
                
                
            }
            
            
            
            if (Math.abs(b - prevB) < 1e-5) {
            	
                converged = true;
            }
        }
        System.out.println("Training complete!");
        double[] wb = Arrays.copyOf(w, w.length + 1);
        wb[w.length] = b;
        return wb;
    }

    private double dotProduct(double[] a, double[] b) {
        double result = 0;
        for (int i = 0; i < a.length; i++) {
            result += a[i] * b[i];
        }
        return result;
    }

    public int predict(double[] x, double[] w, double b) {
        double result = dotProduct(w, x) + b;
        return result >= 0 ? 1 : -1;
    }

    public double calculateAccuracy(List<double[]> testData, List<Integer> testLabels, double[] w, double b) {
        int correctPredictions = 0;
        for (int i = 0; i < testData.size(); i++) {
            int prediction = predict(testData.get(i), w, b);
            if (prediction == testLabels.get(i)) {
                correctPredictions++;
            }
        }
        return (double) correctPredictions / testData.size();
    }
    
  

    // Confusion  Matrices Calculation
    
    public void printConfusionMatrix(List<double[]> data, List<Integer> labels, double[] w, double b) {
    	
        int[][] matrix = new int[2][2]; // [[TP, FP], [FN, TN]]
        
        
        // for (int i =; i < data.) {
        
        
        for (int i = 0; i < data.size(); i++) {
            int predicted = predict(data.get(i), w, b);
            int actual = labels.get(i);
            if (predicted == 1 && actual == 1) {
                matrix[0][0]++; // True Positive
            } else if (predicted == 1 && actual == -1) {
                matrix[0][1]++; // False Positive
            } else if (predicted == -1 && actual == 1) {
                matrix[1][0]++; // False Negative
            } else {
                matrix[1][1]++; // True Negative
            }
        }
        
        
        System.out.println("Confusion Matrix:");
        System.out.println("TP: " + matrix[0][0] + " FP: " + matrix[0][1]);
        System.out.println("FN: " + matrix[1][0] + " TN: " + matrix[1][1]);
    }

    // Cross-validation function for hyperparameter tuning
    public void crossValidate(int folds, double C, double learningRate, int maxIterations) {
        int foldSize = trainingData.size() / folds;
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < trainingData.size(); i++) {
            indices.add(i);
        }

        double totalAccuracy = 0;
        for (int fold = 0; fold < folds; fold++) {
            // Split data into training and test sets for this fold
            List<double[]> trainData = new ArrayList<>();
            List<Integer> trainLabels = new ArrayList<>();
            List<double[]> testData = new ArrayList<>();
            List<Integer> testLabels = new ArrayList<>();

            for (int i = 0; i < indices.size(); i++) {
                if (i >= fold * foldSize && i < (fold + 1) * foldSize) {
                    testData.add(trainingData.get(indices.get(i)));
                    testLabels.add(trainingLabels.get(indices.get(i)));
                } else {
                    trainData.add(trainingData.get(indices.get(i)));
                    trainLabels.add(trainingLabels.get(indices.get(i)));
                }
            }

            // Train SVM on this fold's training data
            List<double[]> tempTrainingData = new ArrayList<>(trainData);
            List<Integer> tempTrainingLabels = new ArrayList<>(trainLabels);
            double[] wb = trainLinearSVM(C, learningRate, maxIterations);

            // Evaluate the model on the test data
            double[] w = Arrays.copyOf(wb, wb.length - 1);
            double b = wb[wb.length - 1];
            double accuracy = calculateAccuracy(testData, testLabels, w, b);
            totalAccuracy += accuracy;

            System.out.println("Fold " + (fold + 1) + " Accuracy: " + accuracy);
        }

        double averageAccuracy = totalAccuracy / folds;
        System.out.println("Average Accuracy across " + folds + " folds: " + averageAccuracy);
    }

    // ----------------- MLP Implementation ------------------
    
    class MLP {
        private int inputSize;
        private int hiddenSize;
        private int outputSize;
        private double[][] W1, W2;
        private double[] b1, b2;
        private double learningRate;

        public MLP(int inputSize, int hiddenSize, int outputSize, double learningRate) {
            this.inputSize = inputSize;
            this.hiddenSize = hiddenSize;
            this.outputSize = outputSize;
            this.learningRate = learningRate;

            // Initialize weights and biases
            this.W1 = new double[inputSize][hiddenSize];
            this.W2 = new double[hiddenSize][outputSize];
            this.b1 = new double[hiddenSize];
            this.b2 = new double[outputSize];

            initializeWeights();
        }

        private void initializeWeights() {
            Random rand = new Random();
            for (int i = 0; i < inputSize; i++) {
                for (int j = 0; j < hiddenSize; j++) {
                	
                	// Small random values
                    W1[i][j] = rand.nextDouble() * 0.01; 
                }
            }

            for (int i = 0; i < hiddenSize; i++) {
                for (int j = 0; j < outputSize; j++) {
                    W2[i][j] = rand.nextDouble() * 0.01;
                }
            }

            for (int i = 0; i < hiddenSize; i++) {
                b1[i] = 0;
            }

            for (int i = 0; i < outputSize; i++) {
                b2[i] = 0;
            }
        }

        private double[] sigmoid(double[] z) {
            double[] result = new double[z.length];
            for (int i = 0; i < z.length; i++) {
                result[i] = 1 / (1 + Math.exp(-z[i]));
            }
            return result;
        }

        private double[] sigmoidDerivative(double[] z) {
            double[] result = new double[z.length];
            for (int i = 0; i < z.length; i++) {
                result[i] = z[i] * (1 - z[i]);
            }
            return result;
        }

        private double[] forward(double[] X) {
            // Forward pass
            double[] hiddenInput = new double[hiddenSize];
            double[] hiddenOutput = new double[hiddenSize];
            
            for (int j = 0; j < hiddenSize; j++) {
                hiddenInput[j] = 0;
                for (int i = 0; i < inputSize; i++) {
                    hiddenInput[j] += X[i] * W1[i][j];
                }
                hiddenInput[j] += b1[j];
                hiddenOutput[j] = sigmoid(hiddenInput)[j];
            }

            double[] outputInput = new double[outputSize];
            double[] output = new double[outputSize];
            for (int j = 0; j < outputSize; j++) {
                outputInput[j] = 0;
                for (int i = 0; i < hiddenSize; i++) {
                    outputInput[j] += hiddenOutput[i] * W2[i][j];
                }
                outputInput[j] += b2[j];
                output[j] = sigmoid(outputInput)[j];
            }

            return output;
        }

        public void train(List<double[]> trainData, List<Integer> trainLabels, int epochs) {
            for (int epoch = 0; epoch < epochs; epoch++) {
                for (int i = 0; i < trainData.size(); i++) {
                    double[] X = trainData.get(i);
                    int y = trainLabels.get(i);

                    // Forward pass
                    double[] output = forward(X);

                    // Backward pass
                    // Compute gradients and update weights here
                    
                    // (This part would involve computing loss and using backpropagation
                    // to update the weights W1, W2, and biases b1, b2)
                }
            }
        }

        
        //public int 
        
        // Prediction logic
        public int predict(double[] X) {
            double[] output = forward(X);
            return (output[0] > 0.5) ? 1 : -1;  // Assuming binary clasification
        }
    }
    
    public static void main(String[] args) {
        Main classifier = new Main();

        String dataSet1 = "/Users/chaitanyasharma/eclipse-workspace/cw_machinelearning/src/machinelearning/dataSet1.csv";
        String dataSet2 = "/Users/chaitanyasharma/eclipse-workspace/cw_machinelearning/src/machinelearning/dataSet2.csv";
        
        // Load training data
        classifier.loadData(dataSet1, false); 
        
        // Load test data
        classifier.loadData(dataSet2, true); 

        // Preprocess Data
        
        System.out.println("Preprocessing data...");
        classifier.encodeLabels();
        classifier.balanceClasses();

        // Apply Polynomial Features
        
        classifier.applyPolynomialFeatures(2);

        // Standardize Data
        
        double[] means = new double[classifier.trainingData.get(0).length];
        double[] stdDevs = new double[classifier.trainingData.get(0).length];
        classifier.standardizeData(classifier.trainingData, means, stdDevs);

        // Perform Cross-Validation
        
        classifier.crossValidate(5, 1.0, 0.001, 5000);

        // Train SVM on the full training data
        
        double[] wb = classifier.trainLinearSVM(1.0, 0.001, 5000);
        double[] w = Arrays.copyOf(wb, wb.length - 1);
        double b = wb[wb.length - 1];

        // Evaluate on Test Data
        
        System.out.println("Evaluating on Test Data...");
        double accuracy = classifier.calculateAccuracy(classifier.testData, classifier.testLabels, w, b);
        System.out.println("Test Accuracy: " + accuracy);

        // Confusion Matrix
        classifier.printConfusionMatrix(classifier.testData, classifier.testLabels, w, b);

        // ---- MLP Evaluation ----
        MLP mlp = classifier.new MLP(classifier.featureCount, 10, 1, 0.001);
        mlp.train(classifier.trainingData, classifier.trainingLabels, 500);
    }
}
