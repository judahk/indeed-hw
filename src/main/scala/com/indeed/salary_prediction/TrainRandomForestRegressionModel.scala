package com.indeed.salary_prediction

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.mllib.tree.{RandomForest, DecisionTree}
import org.apache.spark.{SparkContext, SparkConf}

/**
 * Created by judahk on 11/10/2015.
 */
object TrainRandomForestRegressionModel {
  def main(args: Array[String]) {
    val dataFile = args(0)
    val labelFile = args(1)
    val doFeatureScaling:Boolean = args(2).toBoolean
    val doModelSelection = args(3).toBoolean

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
    if (doFeatureScaling) {
      println("Doing feature scaling")
      val scaler = new StandardScaler(withMean = true, withStd = true).fit(trainingData.map(x => x.features))
      val scaledData = trainingData.map(x => (x.label, scaler.transform(x.features).toArray))
      //    scaledData.take(10).foreach{case(label, feats) =>
      //      feats.foreach(x => print(x + " "))
      //      print(" -- ")
      //      print(label + " ")
      //      println()
      //    }
      trainingData = scaledData.map { case (label, feats) => LabeledPoint(label, Vectors.dense(feats)) }
      println("Done feature scaling")
    }

    // Evaluate decision tree model via CV
    // Define some constants
    val categoricalFeaturesInfo = Map[Int, Int]() // No nominal features as we transform them to numeric
    var numTrees = 20
    val featureSubsetStrategy = "auto"
    val impurity = "variance" // "variance" for regression problems
    var maxDepth = 18
    val maxBins = 32
    val numFolds = 10

    val splits = TrainUtils.createKFoldCVSplits(trainingData, numFolds, sc)

    if (doModelSelection){
      println("Doing model selection")
      val numTreesParams = Range(12,21,2).toList
      val maxDepthParams = Range(12,21,2).toList
      val params = (for(x <- numTreesParams; y <- maxDepthParams) yield (x,y))
      val perf = params.map{t =>
        println("Trying parameter values = " + t.toString)
        val (numTrees, maxDepth) = t
        val res = splits.map{case(train, validation) =>
          // Building the model
          val model = RandomForest.trainRegressor(train, categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)

          // Evaluate model on training examples and compute training error
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
      // Take correlation as perf measure
      val corrPerf = perf.map{case(t,corr,_,_) => (t,corr)}
      val (t, bestCorr) = corrPerf.maxBy(_._2)
      numTrees = t._1
      maxDepth = t._2
      println("Best param values = " + t.toString)
      println("Done model selection")
    }

    println("Doing CV based model evaluation with best param values")
    val result = splits.map{case(train, validation) =>
      // Building the model
      val model = RandomForest.trainRegressor(train, categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)

      // Evaluate model on training examples and compute training error
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

    // Train on whole data
    val modelFullData = RandomForest.trainRegressor(trainingData, categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)

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

    sc.stop()
  }
}
