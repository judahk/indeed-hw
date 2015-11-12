package com.indeed.salary_prediction

import org.apache.hadoop.mapred.FileAlreadyExistsException
import org.apache.log4j.{Logger, Level}
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.{LassoWithSGD, RidgeRegressionWithSGD, LinearRegressionWithSGD, LabeledPoint}
import org.apache.spark.mllib.stat.Statistics

/**
 * Created by judahk on 11/7/2015.
 */
object TrainLinearRegressionModel {
  def main(args: Array[String]) {
    val dataFile = args(0)
    val labelFile = args(1)
    val doFeatureScaling:Boolean = args(2).toBoolean
    val doModelSelection = args(3).toBoolean
    val modelPath = args(4)
    val predictLabelsOnTestData = args(5).toBoolean
    val testDataFile = args(6)
    val outputFile = args(7)
    val printModel = false

    // Turn off Spark logging
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    // Create SPARK conf and context
    val conf = new SparkConf().setAppName("LinearRegressionTest").setMaster("local[*]").set("spark.local.dir", "C:\\tmp")
    //    conf.set("spark.eventLog.enabled", "true")
    //    conf.set("spark.eventLog.dir", "C:\\tmp\\eventLog.out")
    val sc = new SparkContext(conf)

    // Load and modify the data to appropriate form for learning
    println("Loading and creating training data")
    val trainingDataWithJobIDs = TrainUtils.loadTrainingData(dataFile, labelFile, sc)
    var trainingData = trainingDataWithJobIDs.map(t => t._2).cache()
    println("Done loading and creating training data")

    // Scale features using StandardScaler
    if (doFeatureScaling){
      println("Doing feature scaling")
      val scaler = new StandardScaler(withMean = true, withStd = true).fit(trainingData.map(x => x.features))
      val scaledData = trainingData.map(x => (x.label, scaler.transform(x.features).toArray))
      //    scaledData.take(10).foreach{case(label, feats) =>
      //      feats.foreach(x => print(x + " "))
      //      print(" -- ")
      //      print(label + " ")
      //      println()
      //    }
      trainingData = scaledData.map{case(label, feats) => LabeledPoint(label, Vectors.dense(feats))}
      println("Done feature scaling")
    }

    // Evaluate linear regression model via CV
    // Define some constants
    var numIterations = 200 // Set to best value found via CV
    var stepSize = 1.0 // Set to best value found via CV
    val numFolds = 10

    val splits = TrainUtils.createKFoldCVSplits(trainingData, numFolds, sc)

    if (doModelSelection){
      println("Doing model selection")
      val numIterationsParams = Range(100,1000,100).toList
      val stepSizeParams = List(1e-4, 1e-3, 1e-2, 1e-1, 1.0)
      val params = (for(x <- numIterationsParams; y <- stepSizeParams) yield (x,y))
      val perf = params.map{t =>
        println("Trying parameter values = " + t.toString)
        val (numIterations, stepSize) = t
        val res = splits.map{case(train, validation) =>
          val lrSGD = new LinearRegressionWithSGD().setIntercept(true)
          lrSGD.optimizer.setNumIterations(numIterations).setStepSize(stepSize)
          val model = lrSGD.run(train)
//          val model = LinearRegressionWithSGD.train(train, numIterations, stepSize)
          //    val model = RidgeRegressionWithSGD.train(trainingDataScaled, numIterations, stepSize, 0.01)
          //    val model = LassoWithSGD.train(trainingData, numIterations)

          // Evaluate model on validation examples and compute error
          val valuesAndPreds = validation.map { point =>
            val prediction = model.predict(point.features).round.toDouble
            (point.label, prediction)
          }
          val MSE = valuesAndPreds.map{case(v, p) => math.pow((v - p), 2)}.mean()
          val RMSE = math.sqrt(MSE)

          val MAE = valuesAndPreds.map{case(v, p) => math.abs((v - p))}.mean()

          // Compute correlation coefficient between actual and predictions
          val actuals = valuesAndPreds.map{case(v, p) => v}
          val preds = valuesAndPreds.map{case(v, p) => p}
          val correlation: Double = Statistics.corr(actuals, preds, "pearson")
          (correlation,MAE,RMSE)
        }
        val correlation = res.map(t => t._1).reduce((x,y) => x+y)/splits.length
        val MAE = res.map(t => t._2).reduce((x,y) => x+y)/splits.length
        val RMSE = res.map(t => t._3).reduce((x,y) => x+y)/splits.length
        (t,correlation,MAE,RMSE)
      }
      // Take correlation as performance measure to optimize (Try other measures as well)
      val corrPerf = perf.map{case(t,corr,_,_) => (t,corr)}
      val (t, bestCorr) = corrPerf.maxBy(_._2)
      numIterations = t._1
      stepSize = t._2
      println("Best param values = " + t.toString)
      println("Best correlation coefficient = " + bestCorr)
      println("Done model selection")
    }

    println("Doing CV based model evaluation with best param values")
    val result = splits.map{case(train, validation) =>
      // Building the model
      val lrSGD = new LinearRegressionWithSGD().setIntercept(true)
      lrSGD.optimizer.setNumIterations(numIterations).setStepSize(stepSize)
      val model = lrSGD.run(train)
//      val model = LinearRegressionWithSGD.train(train, numIterations, stepSize)
      //    val model = RidgeRegressionWithSGD.train(trainingDataScaled, numIterations, stepSize, 0.01)
      //    val model = LassoWithSGD.train(trainingData, numIterations)

      // Evaluate model on validation examples and compute error
      val valuesAndPreds = validation.map { point =>
        val prediction = model.predict(point.features).round.toDouble
        (point.label, prediction)
      }
      val MSE = valuesAndPreds.map{case(v, p) => math.pow((v - p), 2)}.mean()
      val RMSE = math.sqrt(MSE)

      val MAE = valuesAndPreds.map{case(v, p) => math.abs((v - p))}.mean()

      // Compute correlation coefficient between actual and predictions
      val actuals = valuesAndPreds.map{case(v, p) => v}
      val preds = valuesAndPreds.map{case(v, p) => p}
      val correlation: Double = Statistics.corr(actuals, preds, "pearson")
      (correlation,MAE,RMSE)
    }
    println("Done CV based model evaluation")

    val correlation = result.map(t => t._1).reduce((x,y) => x+y)/splits.length
    val MAE = result.map(t => t._2).reduce((x,y) => x+y)/splits.length
    val RMSE = result.map(t => t._3).reduce((x,y) => x+y)/splits.length

    println("Printing final results:")
    println("10-Fold CV Correlation Coefficient = " + correlation)
    println("10-Fold CV Mean Absolute Error = " + MAE)
    println("10-Fold CV Root Mean Squared Error = " + RMSE)

    // Train model on entire data
    val lrSGD = new LinearRegressionWithSGD().setIntercept(true)
    lrSGD.optimizer.setNumIterations(numIterations).setStepSize(stepSize)
    val modelFullData = lrSGD.run(trainingData)

    // Evaluate model on training examples and compute training error
    val valuesAndPredsTraining = trainingData.map { point =>
      val prediction = modelFullData.predict(point.features).round.toDouble
      (point.label, prediction)
    }
    val TrainingMSE = valuesAndPredsTraining.map{case(v, p) => math.pow((v - p), 2)}.mean()
    val TrainingRMSE = math.sqrt(TrainingMSE)

    val TrainingMAE = valuesAndPredsTraining.map{case(v, p) => math.abs((v - p))}.mean()

    // Compute correlation coefficient between actual and predictions
    val actualsTraining = valuesAndPredsTraining.map{case(v, p) => v}
    val predsTraining = valuesAndPredsTraining.map{case(v, p) => p}
    val TrainingCorrelation: Double = Statistics.corr(actualsTraining, predsTraining, "pearson")

    println("Printing training data results:")
    println("Training Correlation Coefficient = " + TrainingCorrelation)
    println("Training Mean Absolute Error = " + TrainingMAE)
    println("Training Root Mean Squared Error = " + TrainingRMSE)

    if (printModel){
      println("Printing model:")
      println("intercept = " + modelFullData.intercept)
      println("weights = " + modelFullData.weights.toString)
    }

    // Print predictions on training data
    val trainingDataJobIDs = trainingDataWithJobIDs.map(t => t._1)
    val predsTrainingWithJobIDs = trainingDataJobIDs.zip(predsTraining)
    println("Printing predicted labels on training data")
    predsTrainingWithJobIDs.foreach(t => println(t._1 + "," + t._2))
    println("Done printing predicted labels on training data")

    // Save trained model
    try{
      println("Saving trained model")
      modelFullData.save(sc, modelPath)
      println("Done saving trained model")
    }catch {
      case e: FileAlreadyExistsException =>{
        println("Exception in saving model: file already exists")
      }
      case unknown:Throwable => println("Unknown exception occurred while saving model: " + unknown)
    }

    // Predict test data
    if (predictLabelsOnTestData){
      val testDataWithJobIDs = TrainUtils.loadTestData(testDataFile, sc)
      val testDataJobIDs = testDataWithJobIDs.map(t => t._1)
      var testData = testDataWithJobIDs.map(t => t._2)
      // Scale test data
      if (doFeatureScaling){
        println("Doing feature scaling")
        val scaler = new StandardScaler(withMean = true, withStd = true).fit(testData.map(x => x.features))
        val scaledData = testData.map(x => (x.label, scaler.transform(x.features).toArray))
        //    scaledData.take(10).foreach{case(label, feats) =>
        //      feats.foreach(x => print(x + " "))
        //      print(" -- ")
        //      print(label + " ")
        //      println()
        //    }
        testData = scaledData.map{case(label, feats) => LabeledPoint(label, Vectors.dense(feats))}
        println("Done feature scaling")
      }
      val predictedLabels = testData.map { point =>
        val prediction = modelFullData.predict(point.features).round.toDouble
        prediction
      }
      val predictedLabelsWithJobIDs = testDataJobIDs.zip(predictedLabels)
      println("Printing predicted labels on test data")
      predictedLabelsWithJobIDs.foreach(t => println(t._1 + "," + t._2))
      println("Done printing predicted labels on test data")

      println("Saving predicted labels on test data")
      predictedLabelsWithJobIDs.map(t => t._1 + "," + t._2).saveAsTextFile(outputFile)
      println("Done saving predicted labels on test data")
    }

    sc.stop()
  }
}