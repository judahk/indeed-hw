package com.indeed.salary_prediction;

import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.unsupervised.attribute.NominalToBinary;

import java.io.File;
import java.util.Random;

/**
 * Created by judahk on 11/8/2015.
 */
public class TrainWekaLinearRegression {
  public static void main(String[] args) throws Exception {
    // Read csv data file
    String dataFile = args[0];

    DataSource source = new DataSource(dataFile);
    Instances data = source.getDataSet();
    data.deleteAttributeAt(0); // Remove job id field
    data.setClassIndex(data.numAttributes() - 1);
//    printInstances(data);

    // Change nominal features to numeric via dummy variables
    data = changeNominalToNumeric(data);
//    printInstances(data);

    // 10-fold CV using Evaluation class
    LinearRegression lr = new LinearRegression();
    SelectedTag tag = new SelectedTag(LinearRegression.SELECTION_NONE, LinearRegression.TAGS_SELECTION);
    lr.setAttributeSelectionMethod(tag); // turn off attribute selection
    lr.setEliminateColinearAttributes(false);
    Evaluation eval = new Evaluation(data);
    eval.crossValidateModel(lr, data, 10, new Random(1));
    System.out.println("10-Fold CV Correlation Coefficient = " + Double.toString(eval.correlationCoefficient()));
    System.out.println("10-Fold CV Mean Absolute Error = " + Double.toString(eval.meanAbsoluteError()));
    System.out.println("10-Fold CV Root Mean Squared Error = " + Double.toString(eval.rootMeanSquaredError()));

//    int folds = 10;
//    double sumCorrCV = 0.0;
//    double sumMAECV = 0.0;
//    double sumRMSECV = 0.0;
//    for (int fold = 0; fold < folds; fold++) {
//      System.out.println("Running Fold: " + fold);
//      int seed = fold+1;  // every run gets a new, but defined seed value
//      Random rand = new Random(seed);   // create seeded number generator
//      Instances randData = new Instances(data);   // create copy of original data
//      randData.randomize(rand);         // randomize data with number generator
//
//      Instances train = randData.trainCV(folds, fold);
//      Instances test = randData.testCV(folds, fold);
//
//      // Building the model
//      LinearRegression lr = new LinearRegression();
//      SelectedTag tag = new SelectedTag(LinearRegression.SELECTION_NONE, LinearRegression.TAGS_SELECTION);
//      lr.setAttributeSelectionMethod(tag); // turn off attribute selection
//      lr.setEliminateColinearAttributes(false);
//      lr.buildClassifier(train);
//
//      // Evaluate model on validation examples and compute validation error
//      double sumMSE = 0.0;
//      double sumMAE = 0.0;
//      double[] actuals = new double[data.numInstances()];
//      double[] preds = new double[data.numInstances()];
//      for (int j=0; j < test.numInstances(); j++){
//        Instance inst = test.instance(j);
//        double prediction = lr.classifyInstance(inst);
//        preds[fold] = prediction;
//        double actual = inst.classValue();
//        actuals[fold] = actual;
//        sumMSE += Math.pow((prediction-actual), 2);
//        sumMAE += Math.abs(prediction-actual);
//      }
//      double MSE = (1.*sumMSE)/data.numInstances();
//      double RMSE = Math.sqrt(MSE);
//      sumRMSECV += RMSE;
//      double MAE = (1.*sumMAE)/data.numInstances();
//      sumMAECV += MAE;
//      // Compute correlation coefficient between actual and predictions
//      PearsonsCorrelation pearsonsCorrelation = new PearsonsCorrelation();
//      double correlation = pearsonsCorrelation.correlation(actuals, preds);
//      sumCorrCV += correlation;
//      System.out.println("Fold " + fold + " Correlation Coefficient = " + correlation);
//      System.out.println("Fold " + fold + " Mean Absolute Error = " + MAE);
//      System.out.println("Fold " + fold + " Root Mean Squared Error = " + RMSE);
//    }
//    double correlation = sumCorrCV/folds;
//    double MAE = sumMAECV/folds;
//    double RMSE = sumRMSECV/folds;
//    System.out.println("10-Fold CV Correlation Coefficient = " + correlation);
//    System.out.println("10-Fold CV Mean Absolute Error = " + MAE);
//    System.out.println("10-Fold CV Root Mean Squared Error = " + RMSE);

    // Output model
//    System.out.println(lr.toString());
  }

  public static Instances changeNominalToNumeric(Instances data) throws Exception{
    NominalToBinary filter = new NominalToBinary();
    filter.setInputFormat(data);
    filter.setBinaryAttributesNominal(false);
    filter.setTransformAllValues(false);
    for (int i = 0; i < data.numInstances(); i++) {
      filter.input(data.instance(i));
    }
    filter.batchFinished();
    Instances newData = filter.getOutputFormat();
    Instance processed;
    while ((processed = filter.output()) != null) {
      newData.add(processed);
    }
    return newData;
  }

  public static void printInstances(Instances data){
    for (int i=0; i < data.numInstances(); i++){
      Instance inst = data.instance(i);
      System.out.println(inst.toString());
    }
  }
}
