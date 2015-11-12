package com.indeed.salary_prediction

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

/**
 * Created by judahk on 11/9/2015.
 */
object TrainUtils {
  def loadTrainingData(dataFile: String, labelFile: String, sc: SparkContext) : RDD[(String, LabeledPoint)] = {
    val featureData = loadFeatureData(dataFile, sc)
    val labels = loadLabelData(labelFile, sc)

    // Change nominal features to numeric via dummy variables
    var dummyFeats = scala.collection.mutable.Map[Int, scala.collection.mutable.Map[String, Int]]()
    var dummyFeatsCount = scala.collection.mutable.Map[Int, Int]()
    var totalDummyVars = 0 // count how many dummy vars we are introducing
    for (indx <- 0 to 4){ //try to remove hardcoded indices
      val vals = featureData.map{case(_,array) => array(indx)}
      val uniqueVals = vals.distinct().collect() // distinct values can be collected as they are not very large often
      val levels = uniqueVals.length
      val numDummyFeats = levels - 1
      val uniqueValsWithIds = uniqueVals.zipWithIndex
      var indices = scala.collection.mutable.Map[String, Int]()
      uniqueValsWithIds.foreach(t => indices += (t._1 -> t._2.toInt))
      dummyFeats += (indx -> indices)
      dummyFeatsCount += (indx -> numDummyFeats)
      totalDummyVars = totalDummyVars + numDummyFeats
    }
    val newData = featureData.map{case(jobID, dataArr) =>
      val array = dataArr.zipWithIndex.flatMap{t => t._2 match {
        case indx if (0 <= indx && indx <= 4) =>
          var arr = new Array[Double](dummyFeatsCount(indx))
          val v = dummyFeats(indx)(dataArr(indx))
          if (v != 0) arr(v-1) = 1.0
          arr
        case _ =>
          var arr = new Array[Double](1)
          arr(0) = t._1.toDouble
          arr
      }
      }
      (jobID, array)
    }
    //    newData.take(10).foreach{array => array.foreach(x => print(x + " ")) // generally do take(n) instead of collect to see RDD data (to avoid out of memory issues on driver machine)!!
    //      println()
    //    }
    val labelData = labels.map{case(jobID,label) => (jobID,label.toDouble)}
    val data = newData.join(labelData)
    // Create training data (RDD of LabeledPoint)
    val trainingData = data.map {case(jobID,(array,label)) =>
      (jobID, LabeledPoint(label, Vectors.dense(array)))
    }
    trainingData
  }

  def loadTestData(dataFile: String, sc: SparkContext) : RDD[(String, LabeledPoint)] = {
    val featureData = loadFeatureData(dataFile, sc)

    // Change nominal features to numeric via dummy variables
    var dummyFeats = scala.collection.mutable.Map[Int, scala.collection.mutable.Map[String, Int]]()
    var dummyFeatsCount = scala.collection.mutable.Map[Int, Int]()
    var totalDummyVars = 0 // count how many dummy vars we are introducing
    for (indx <- 0 to 4){ //try to remove hardcoded indices
      val vals = featureData.map{case(_,array) => array(indx)}
      val uniqueVals = vals.distinct().collect() // distinct values can be collected as they are not very large often
      val levels = uniqueVals.length
      val numDummyFeats = levels - 1
      val uniqueValsWithIds = uniqueVals.zipWithIndex
      var indices = scala.collection.mutable.Map[String, Int]()
      uniqueValsWithIds.foreach(t => indices += (t._1 -> t._2.toInt))
      dummyFeats += (indx -> indices)
      dummyFeatsCount += (indx -> numDummyFeats)
      totalDummyVars = totalDummyVars + numDummyFeats
    }
    val newData = featureData.map{case(jobID, dataArr) =>
      val array = dataArr.zipWithIndex.flatMap{t => t._2 match {
        case indx if (0 <= indx && indx <= 4) =>
          var arr = new Array[Double](dummyFeatsCount(indx))
          val v = dummyFeats(indx)(dataArr(indx))
          if (v != 0) arr(v-1) = 1.0
          arr
        case _ =>
          var arr = new Array[Double](1)
          arr(0) = t._1.toDouble
          arr
      }
      }
      (jobID, array)
    }
    //    newData.take(10).foreach{array => array.foreach(x => print(x + " ")) // generally do take(n) instead of collect to see RDD data (to avoid out of memory issues on driver machine)!!
    //      println()
    //    }

    // Create test data (RDD of LabeledPoint)
    val testData = newData.map {case(jobID,array) =>
      (jobID, LabeledPoint(0.0, Vectors.dense(array)))
    }
    testData
  }

  def loadFeatureData(dataFile: String, sc: SparkContext) : RDD[(String, Array[String])] = {
    val data = sc.textFile(dataFile)
    // Skip header line
    val dataNoHeader = data.mapPartitionsWithIndex {(i, iterator) =>
      if (i == 0 && iterator.hasNext){
        iterator.next()
        iterator
      }else
        iterator
    }
    val newData = dataNoHeader.map{l =>
      val col = l.split(",")
      val jobID = col(0)
      val len = col.length
      val rest = col.takeRight(len-1)
      (jobID, rest)
    }
    newData
  }

  def loadLabelData(dataFile: String, sc: SparkContext) : RDD[(String, String)] = {
    val data = sc.textFile(dataFile)
    // Skip header line
    val dataNoHeader = data.mapPartitionsWithIndex {(i, iterator) =>
      if (i == 0 && iterator.hasNext){
        iterator.next()
        iterator
      }else
        iterator
    }
    val newData = dataNoHeader.map{l =>
      val col = l.split(",")
      val jobID = col(0)
      val label = col(1)
      (jobID, label)
    }
    newData
  }

  def createKFoldCVSplits(data: RDD[LabeledPoint], numFolds: Int, sc: SparkContext) : Array[(RDD[LabeledPoint],RDD[LabeledPoint])] = {
    val wgt = 1.0/numFolds
    val wgts: Array[Double] = new Array[Double](numFolds)
    for (i <- 0 to numFolds-1){
      wgts(i) = wgt
    }
    val partitions = data.randomSplit(wgts)

    val splits = new Array[(RDD[LabeledPoint],RDD[LabeledPoint])](numFolds)
    for (k <- 0 to numFolds-1){
      val testData = partitions(k)
      var trainData:RDD[LabeledPoint] = sc.emptyRDD
      for (l <- 0 to numFolds-1){
        if (l != k)
          trainData = trainData.union(partitions(l))
      }
      val split = (trainData, testData)
      splits(k) = split
    }
    splits
  }

  def printKFoldCVSplits(splits: Array[(RDD[LabeledPoint],RDD[LabeledPoint])]) : Unit = {
    splits.zipWithIndex.foreach{case((train, test),fold) =>
      println("Printing training split of fold = " + fold)
      TrainUtils.printData(train, 10)
      println()
      println("Printing test split of fold = " + fold)
      TrainUtils.printData(test, 10)
      println()
      println()
    }
  }

  def printSingleKFoldCVSplit(splits: Array[(RDD[LabeledPoint],RDD[LabeledPoint])], fold: Int): Unit ={
    val train = splits(fold)._1
    val test = splits(fold)._2
    println("Printing training split of fold = " + fold)
    TrainUtils.printData(train)
    println()
    println("Printing test split of fold = " + fold)
    TrainUtils.printData(test)
    println()
  }

  def printData(data: RDD[LabeledPoint], size: Int = -1) : Unit = size match {
    case -1 =>
      data.take(data.count.toInt).foreach{lp =>
        println(lp.toString)
      }
    case _ =>
      data.take(size).foreach{lp =>
        println(lp.toString)
      }
  }

  def main(args: Array[String]) {
    val dataFile = args(0)
    val labelFile = args(1)

    // Create SPARK conf and context
    val conf = new SparkConf().setAppName("CrossValidationTest").setMaster("local[*]").set("spark.local.dir", "C:\\tmp")
    //    conf.set("spark.eventLog.enabled", "true")
    //    conf.set("spark.eventLog.dir", "C:\\tmp\\eventLog.out")
    val sc = new SparkContext(conf)

    // Load and modify the data to appropriate form for learning
    println("Loading and creating training data")
    val trainingDataWithJobIDs = loadTrainingData(dataFile, labelFile, sc)
    val trainingData = trainingDataWithJobIDs.map(t => t._2).cache()
    println("Done loading and creating training data")

    val k = 10
    val splits = TrainUtils.createKFoldCVSplits(trainingData, k, sc)

    // Print cv splits
//    TrainUtils.printKFoldCVSplits(splits)

    // Print single split
    TrainUtils.printSingleKFoldCVSplit(splits, 0)
  }
}
