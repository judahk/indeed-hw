package com.indeed.salary_prediction

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.{LinearRegressionWithSGD, LabeledPoint}
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.{SparkContext, SparkConf}

/**
 * Created by judahk on 11/8/2015.
 */
object TrainDecisionTreeRegression {
  def main(args: Array[String]) {
    val dataFile = args(0)
    val doFeatureScaling:Boolean = args(1).toBoolean
    val doModelSelection = args(2).toBoolean

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
    val data = sc.textFile(dataFile)

    // Skip header line
    val dataNoHeader = data.mapPartitionsWithIndex {(i, iterator) =>
      if (i == 0 && iterator.hasNext){
        iterator.next()
        iterator
      }else
        iterator
    }

    // Remove job_id field as it is not predictive
    val dataNoJobId = dataNoHeader.map{l =>
      val col = l.split(",")
      val len = col.length
      val rest = col.takeRight(len-1)
      rest
    }

    // Change nominal features to numeric via dummy variables
    var dummyFeats = scala.collection.mutable.Map[Int, scala.collection.mutable.Map[String, Int]]()
    var dummyFeatsCount = scala.collection.mutable.Map[Int, Int]()
    var totalDummyVars = 0 // count how many dummy vars we are introducing
    for (indx <- 0 to 4){ //try to remove hardcoded indices
    val vals = dataNoJobId.map(array => array(indx))
      val uniqueVals = vals.distinct().collect() // distinct values can be collected as they are not very large often
      val levels = uniqueVals.length
      val numDummyFeats = levels - 1
      val uniqueValsWithIds = uniqueVals.zipWithIndex
      var indices = scala.collection.mutable.Map[String, Int]()
      uniqueValsWithIds.foreach(args => indices += (args._1 -> args._2.toInt))
      dummyFeats += (indx -> indices)
      dummyFeatsCount += (indx -> numDummyFeats)
      totalDummyVars = totalDummyVars + numDummyFeats
    }
    val newData = dataNoJobId.map{dataArr =>
      dataArr.zipWithIndex.flatMap{args => args._2 match {
        case indx if (0 <= indx && indx <= 4) =>
          var arr = new Array[Double](dummyFeatsCount(indx))
          val v = dummyFeats(indx)(dataArr(indx))
          if (v != 0) arr(v-1) = 1.0
          arr
        case _ =>
          var arr = new Array[Double](1)
          arr(0) = args._1.toDouble
          arr
      }
      }
    }
    //    newData.take(10).foreach{array => array.foreach(x => print(x + " ")) // generally do take(n) instead of collect to see RDD data (to avoid out of memory issues on driver machine)!!
    //      println()
    //    }

    var trainingData = newData.map { array =>
      LabeledPoint(array(array.length-1), Vectors.dense(array.take(array.length-1)))
    }.cache()
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
    val impurity = "variance" // "variance" for regression problems
    var maxDepth = 12
    var maxBins = 16
    val numFolds = 10

    val splits = TrainUtils.createKFoldCVSplits(trainingData, numFolds, sc)

    if (doModelSelection){
      println("Doing model selection")
      val depthParams = Range(5,30,1).toList
      val binsParams = List(16, 20, 22, 24, 26, 28, 30, 32, 34, 36, 40)
      val params = for (x <- depthParams; y <- binsParams) yield (x,y)
      val perf = params.map{args =>
        println("Trying parameter values = " + args.toString)
        val (maxDepth, maxBins) = args
        val res = splits.map{case(train, test) =>
          // Building the model
          val model = DecisionTree.trainRegressor(train, categoricalFeaturesInfo, impurity, maxDepth, maxBins)

          // Evaluate model on training examples and compute training error
          val valuesAndPreds = test.map { point =>
            val prediction = model.predict(point.features)
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
        val correlation = res.map(args => args._1).reduce((x,y) => x+y)/splits.length
        val MAE = res.map(args => args._2).reduce((x,y) => x+y)/splits.length
        val RMSE = res.map(args => args._3).reduce((x,y) => x+y)/splits.length
        (args,correlation,MAE,RMSE)
      }
      // Take correlation as perf measure
      val corrPerf = perf.map{case(args,corr,_,_) => (args,corr)}
      val (args, bestCorr) = corrPerf.maxBy(_._2)
      maxDepth = args._1
      maxBins = args._2
      println("Best param values = " + args.toString)
      println("Done model selection")
    }

    println("Doing CV based model evaluation with best param values")
    val result = splits.map{case(train, test) =>
      // Building the model
      val model = DecisionTree.trainRegressor(train, categoricalFeaturesInfo, impurity, maxDepth, maxBins)

      // Evaluate model on training examples and compute training error
      val valuesAndPreds = test.map { point =>
        val prediction = model.predict(point.features)
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

    val correlation = result.map(args => args._1).reduce((x,y) => x+y)/splits.length
    val MAE = result.map(args => args._2).reduce((x,y) => x+y)/splits.length
    val RMSE = result.map(args => args._3).reduce((x,y) => x+y)/splits.length

    println("Printing final results:")
    println("10-Fold CV Correlation Coefficient = " + correlation)
    println("10-Fold CV Mean Absolute Error = " + MAE)
    println("10-Fold CV Root Mean Squared Error = " + RMSE)

    // Train on whole data
    val modelFullData = DecisionTree.trainRegressor(trainingData, categoricalFeaturesInfo, impurity, maxDepth, maxBins)

    // Evaluate model on training examples and compute training error
    val valuesAndPredsTraining = trainingData.map { point =>
      val prediction = modelFullData.predict(point.features)
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