// Databricks notebook source
import co.wiklund.disthist._
import SpatialTreeFunctions._
import MergeEstimatorFunctions._
import HistogramFunctions._
import LeafMapFunctions._
import MDEFunctions._
import Types._

import org.apache.spark.mllib.random.RandomRDDs.uniformVectorRDD
import org.apache.spark.mllib.linalg.Vectors

import spark.implicits._

// COMMAND ----------

val numPartitions = 64
spark.conf.set("spark.default.parallelism", numPartitions.toString)

val dimensions = 1000
val sizeExp = 8
val trainSize = math.pow(10, sizeExp).toLong
val testSize = trainSize / 5

val finestResVol = 1.0 / (trainSize * trainSize)
// val finestResSideLength = 1e-3

val rootPath = s"dbfs:/Users/johannes.graner@gmail.com/sparkDensityTree/uniform/${dimensions}d_${sizeExp}n/"
// val rootPath = s"dbfs:/Users/johannes.graner@gmail.com/sparkDensityTree/uniform/${dimensions}d/"
val labeledTrainPath = rootPath + "labeledTrain"
val trainingPath = rootPath + "countedTrain"
val labeledTestPath = rootPath + "labeledTest"
val testPath = rootPath + "countedTest"
val histogramPath = rootPath + "trainedHistogram"
val mdeHistPath = rootPath + "mdeHist"

val rewriteInitial = false
val rewriteMerged = false
val rewriteMDE = false

// COMMAND ----------



// COMMAND ----------

display(dbutils.fs.ls(rootPath))

// COMMAND ----------

val rootBox = Rectangle(
  (1 to dimensions).toVector.map(_ => 0.0),
  (1 to dimensions).toVector.map(_ => 1.0)
)
val tree = widestSideTreeRootedAt(rootBox)
val finestResDepth = tree.descendBoxPrime(Vectors.dense(rootBox.low.toArray)).dropWhile(_._2.volume > finestResVol).head._1.depth
// val finestResDepth = tree.descendBoxPrime(Vectors.dense(rootBox.low.toArray)).dropWhile(_._2.widths.max > finestResSideLength).head._1.depth

// COMMAND ----------

for (sizeExp <- Seq(7, 8)) {
  for (dimensions <- Seq(1, 10, 100, 1000)) {
    val rootPath = s"dbfs:/Users/johannes.graner@gmail.com/sparkDensityTree/uniform/${dimensions}d_${sizeExp}n/"
    val mdeHistPath = rootPath + "mdeHist"
    
    val rootBox = Rectangle(
      (1 to dimensions).toVector.map(_ => 0.0),
      (1 to dimensions).toVector.map(_ => 1.0)
    )
    val tree = widestSideTreeRootedAt(rootBox)

    val mdeHist = {
      val mdeCounts = spark.read.parquet(mdeHistPath).as[(NodeLabel, Count)]
      Histogram(tree, mdeCounts.map(_._2).reduce(_+_), fromNodeLabelMap(mdeCounts.collect.toMap))
    }
    val l1Error = toDensityHistogram(mdeHist).densityMap.vals.map{ case (dens, _) => math.abs(dens - uniformDensity)}.sum
    println(s"sizeExp: ${sizeExp}, dim: ${dimensions},\t leaf count: ${mdeHist.counts.leaves.length},\t error: ${l1Error}")
  }
}

// COMMAND ----------

if (rewriteInitial) {
  dbutils.fs.rm(labeledTrainPath, true)
  dbutils.fs.rm(trainingPath, true)
  val rawTrainRDD = uniformVectorRDD(spark.sparkContext, trainSize, dimensions, numPartitions * dimensions, 1234)
  val labeledTrainRDD = labelAtDepth(tree, finestResDepth, rawTrainRDD)
  spark.createDataset(labeledTrainRDD.map{ case (lab, vec) => (NodeLabel(lab), vec.toArray)}).write.mode("overwrite").parquet(labeledTrainPath)
  val labeledTrainDS = spark.read.parquet(labeledTrainPath).as[(NodeLabel, Array[Double])]
  labeledToCountedDS(labeledTrainDS).write.mode("overwrite").parquet(trainingPath)
  
  dbutils.fs.rm(labeledTestPath, true)
  dbutils.fs.rm(testPath, true)
  val rawTestRDD = uniformVectorRDD(spark.sparkContext, testSize, dimensions, numPartitions * dimensions, 4321)
  val labeledTestRDD = labelAtDepth(tree, finestResDepth, rawTestRDD)
  spark.createDataset(labeledTestRDD.map{ case (lab, vec) => (NodeLabel(lab), vec.toArray)}).write.mode("overwrite").parquet(labeledTestPath)
  val labeledTestDS = spark.read.parquet(labeledTestPath).as[(NodeLabel, Array[Double])]
  labeledToCountedDS(labeledTestDS).write.mode("overwrite").parquet(testPath)
}

// COMMAND ----------

val countedTrainDS = spark.read.parquet(trainingPath).as[(NodeLabel, Count)]
val countedTestDS = spark.read.parquet(testPath).as[(NodeLabel, Count)].cache

// COMMAND ----------

val countLimit = 6500
val stepSize = finestResDepth / 4
val mergedDS = if (rewriteMerged) {
  dbutils.fs.rm(histogramPath, true)
  mergeLeaves(tree, countedTrainDS, countLimit, stepSize, histogramPath, true)
} else
  spark.read.parquet(histogramPath).as[(NodeLabel, Count)]

// COMMAND ----------

val trainedHist = Histogram(tree, mergedDS.map(_._2).reduce(_+_), fromNodeLabelMap(mergedDS.collect.toMap))

// COMMAND ----------

val mdeHist = if (rewriteMDE) {
  val mde = getMDE(trainedHist, countedTestDS, 10, true)
  mde.counts.toIterable.toSeq.toDS.write.mode("overwrite").parquet(mdeHistPath)
  mde
} else {
  val mdeCounts = spark.read.parquet(mdeHistPath).as[(NodeLabel, Count)]
  Histogram(tree, mdeCounts.map(_._2).reduce(_+_), fromNodeLabelMap(mdeCounts.collect.toMap))
}

// COMMAND ----------

val uniformDensity = 1.0
val l1Error = toDensityHistogram(mdeHist).densityMap.vals.map{ case (dens, _) => math.abs(dens - uniformDensity)}.sum

// COMMAND ----------

mdeHist.counts.leaves.length
