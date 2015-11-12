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

    // Train model on entire data

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
