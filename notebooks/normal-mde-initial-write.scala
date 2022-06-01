// Databricks notebook source
import co.wiklund.disthist._
import SpatialTreeFunctions._
import MergeEstimatorFunctions._
import HistogramFunctions._
import LeafMapFunctions._
import MDEFunctions._
import Types._

import org.apache.spark.mllib.random.RandomRDDs.normalVectorRDD
import org.apache.spark.mllib.linalg.Vectors

import spark.implicits._

// COMMAND ----------

// Widgets
val dimensions = dbutils.widgets.get("dimensions").toInt
val sizeExp = dbutils.widgets.get("sizeExp").toInt
val countLimit = dbutils.widgets.get("countLimit").toInt

// COMMAND ----------

val numPartitions = 18 * 4
spark.conf.set("spark.default.parallelism", numPartitions.toString)

// val dimensions = 2
// val sizeExp = 8
val trainSize = math.pow(10, sizeExp).toLong
val testSize = trainSize / 2
// val finestResVolume = 1.0 / (trainSize * trainSize)
val finestResSideLength = 1e-5
val rootPath = s"dbfs:/Users/johannes.graner@gmail.com/sparkDensityTree/normal/${dimensions}d_${sizeExp}n/"
val labeledTrainPath = rootPath + "labeledTrain"
val trainingPath = rootPath + "countedTrain"
val labeledTestPath = rootPath + "labeledTest"
val testPath = rootPath + "countedTest"

val histogramPath = rootPath + "trainedHistogram"
val mdeHistPath = rootPath + "mdeHist"

// COMMAND ----------

val rootBox = Rectangle(
  (1 to dimensions).toVector.map(_ => -10.0),
  (1 to dimensions).toVector.map(_ => 10.0)
)
val tree = widestSideTreeRootedAt(rootBox)
val finestResDepth = tree.descendBoxPrime(Vectors.dense(rootBox.low.toArray)).dropWhile(_._2.widths.max > finestResSideLength).head._1.depth

// val countLimit = 1000
val stepSize = math.ceil(finestResDepth / 8.0).toInt
val kInMDE = 10

// COMMAND ----------

dbutils.fs.rm(labeledTrainPath, true)
dbutils.fs.rm(trainingPath, true)
val rawTrainRDD = normalVectorRDD(spark.sparkContext, trainSize, dimensions, numPartitions * dimensions, 1234)
val labeledTrainRDD = labelAtDepth(tree, finestResDepth, rawTrainRDD)
spark.createDataset(labeledTrainRDD.map{ case (lab, vec) => (NodeLabel(lab), vec.toArray)}).write.mode("overwrite").parquet(labeledTrainPath)
val labeledTrainDS = spark.read.parquet(labeledTrainPath).as[(NodeLabel, Array[Double])]
labeledToCountedDS(labeledTrainDS).write.mode("overwrite").parquet(trainingPath)

dbutils.fs.rm(labeledTestPath, true)
dbutils.fs.rm(testPath, true)
val rawTestRDD = normalVectorRDD(spark.sparkContext, testSize, dimensions, numPartitions * dimensions, 4321)
val labeledTestRDD = labelAtDepth(tree, finestResDepth, rawTestRDD)
spark.createDataset(labeledTestRDD.map{ case (lab, vec) => (NodeLabel(lab), vec.toArray)}).write.mode("overwrite").parquet(labeledTestPath)
val labeledTestDS = spark.read.parquet(labeledTestPath).as[(NodeLabel, Array[Double])]
labeledToCountedDS(labeledTestDS).write.mode("overwrite").parquet(testPath)

// COMMAND ----------

val countedTrainDS = spark.read.parquet(trainingPath).as[(NodeLabel, Count)]
val countedTestDS = spark.read.parquet(testPath).as[(NodeLabel, Count)].cache

dbutils.fs.rm(histogramPath, true)
val mergedDS = mergeLeaves(tree, countedTrainDS, countLimit, stepSize, histogramPath, true)

// COMMAND ----------

val mde = getMDE(
  Histogram(tree, mergedDS.map(_._2).reduce(_+_), fromNodeLabelMap(mergedDS.collect.toMap)), 
  countedTestDS, 
  kInMDE, 
  true
)
mde.counts.toIterable.toSeq.toDS.write.mode("overwrite").parquet(mdeHistPath)
