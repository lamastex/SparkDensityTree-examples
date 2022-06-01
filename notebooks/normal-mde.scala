// Databricks notebook source
import co.wiklund.disthist._
import SpatialTreeFunctions._
import MergeEstimatorFunctions._
import HistogramFunctions._
import LeafMapFunctions._
import MDEFunctions._
import Types._

import org.apache.spark.mllib.linalg.Vectors

import spark.implicits._

// COMMAND ----------

val numPartitions = 64
spark.conf.set("spark.default.parallelism", numPartitions.toString)

val dimensions = 1
val sizeExp = 7
val trainSize = math.pow(10, sizeExp).toLong
val testSize = trainSize / 5

// val finestResVol = 1.0 / (trainSize * trainSize)
val finestResSideLength = 1e-5

val rootPath = s"dbfs:/Users/johannes.graner@gmail.com/sparkDensityTree/normal/${dimensions}d_${sizeExp}n/"
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

display(dbutils.fs.ls(rootPath))

// COMMAND ----------

val rootBox = Rectangle(
  (1 to dimensions).toVector.map(_ => -10.0),
  (1 to dimensions).toVector.map(_ => 10.0)
)
val tree = widestSideTreeRootedAt(rootBox)
// val finestResDepth = tree.descendBoxPrime(Vectors.dense(rootBox.low.toArray)).dropWhile(_._2.volume > finestResVol).head._1.depth
val finestResDepth = tree.descendBoxPrime(Vectors.dense(rootBox.low.toArray)).dropWhile(_._2.widths.max > finestResSideLength).head._1.depth

// COMMAND ----------

// if (rewriteInitial) {
//   dbutils.fs.rm(labeledTrainPath, true)
//   dbutils.fs.rm(trainingPath, true)
//   val rawTrainRDD = uniformVectorRDD(spark.sparkContext, trainSize, dimensions, numPartitions * dimensions, 1234)
//   val labeledTrainRDD = labelAtDepth(tree, finestResDepth, rawTrainRDD)
//   spark.createDataset(labeledTrainRDD.map{ case (lab, vec) => (NodeLabel(lab), vec.toArray)}).write.mode("overwrite").parquet(labeledTrainPath)
//   val labeledTrainDS = spark.read.parquet(labeledTrainPath).as[(NodeLabel, Array[Double])]
//   labeledToCountedDS(labeledTrainDS).write.mode("overwrite").parquet(trainingPath)
  
//   dbutils.fs.rm(labeledTestPath, true)
//   dbutils.fs.rm(testPath, true)
//   val rawTestRDD = uniformVectorRDD(spark.sparkContext, testSize, dimensions, numPartitions * dimensions, 4321)
//   val labeledTestRDD = labelAtDepth(tree, finestResDepth, rawTestRDD)
//   spark.createDataset(labeledTestRDD.map{ case (lab, vec) => (NodeLabel(lab), vec.toArray)}).write.mode("overwrite").parquet(labeledTestPath)
//   val labeledTestDS = spark.read.parquet(labeledTestPath).as[(NodeLabel, Array[Double])]
//   labeledToCountedDS(labeledTestDS).write.mode("overwrite").parquet(testPath)
// }

// COMMAND ----------

val countedTrainDS = spark.read.parquet(trainingPath).as[(NodeLabel, Count)]
val countedTestDS = spark.read.parquet(testPath).as[(NodeLabel, Count)].cache

// COMMAND ----------

countedTrainDS.count

// COMMAND ----------

countedTrainDS.orderBy($"count(1)".desc).first

// COMMAND ----------

val countLimit = 1000
val stepSize = finestResDepth / 4
val mergedDS = if (rewriteMerged) {
  dbutils.fs.rm(histogramPath, true)
  mergeLeaves(tree, countedTrainDS, countLimit, stepSize, histogramPath, true)
} else
  spark.read.parquet(histogramPath).as[(NodeLabel, Count)]

// COMMAND ----------

mergedDS.orderBy('_2.desc).show

// COMMAND ----------

val localMerged = mergedDS.collect.toVector

// COMMAND ----------

def checkAncestries(nodes: Vector[NodeLabel]): Unit = {
  println("checking...")
  nodes.groupBy(_.depth).toVector.sortBy(-_._1).foldLeft[Vector[NodeLabel]](Vector()){ case (acc, curr) =>
    val currDepth = curr._1
    val currNodes = curr._2
    val lastTrunced = acc.map(_.truncate(currDepth))
    if (lastTrunced.intersect(currNodes).length > 0) println(s"intersection at depth $currDepth: ${lastTrunced.intersect(currNodes)} are ancestors to other nodes")
    lastTrunced ++ currNodes
  }
  println("finished!")
}

// COMMAND ----------

val tn: Int => NodeLabel = NodeLabel(_)
checkAncestries(Vector(2,3,4,8,9,31) map tn)

// COMMAND ----------

checkAncestries(localMerged.unzip._1)

// COMMAND ----------

println(mergedDS.count)
mergedDS.map{ case (node, count) => (node.truncate(3), count) }.groupBy('_1).sum("_2").orderBy('_1).show

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

mdeHist.counts.leaves.map(node => (node, tree.volumeAt(node))).maxBy(_._2)

// COMMAND ----------

spark.read.parquet(mdeHistPath).as[(NodeLabel, Count)].orderBy('_1).show

// COMMAND ----------

mdeHist.counts.leaves.length

// COMMAND ----------

val margHist = marginalize(toDensityHistogram(mdeHist), Vector(0))

// COMMAND ----------

margHist.densityMap.leaves == toDensityHistogram(mdeHist).densityMap.leaves

// COMMAND ----------

display(margHist.densityMap.toIterable.map{ case (node, (prob, vol)) => (margHist.tree.cellAt(node).centre(0), prob * vol)}.toVector.toDS)

// COMMAND ----------

import math.{ abs, exp, sqrt, atan }

// COMMAND ----------

val PI = atan(1)*4

def normDens(x: Double): Double =
 exp(-x*x/2) / sqrt(2*PI)

def normDensND(xs: Vector[Double]): Double = {
  val dim = xs.length
  exp(xs.map(x => -x*x/2).sum) / Stream.continually(sqrt(2*PI)).take(dim).reduce(_*_)
}
  

// COMMAND ----------

def midPoint(rec: Rectangle): Vector[Double] = {
  (0 until rec.dimension).map(rec.centre(_)).toVector
}

def corners(rec: Rectangle): Vector[Vector[Double]] =
  (1 until rec.dimension).toVector
    .foldLeft(Vector(Vector(rec.low(0)), Vector(rec.high(0)))){ case (acc, dim) => 
      acc.flatMap(corner => Vector(corner :+ rec.low(dim), corner :+ rec.high(dim)))
    }

// COMMAND ----------

val rec = Rectangle(Vector(0,0,0),Vector(1,1,1))

// COMMAND ----------

midPoint(rec)

// COMMAND ----------

corners(rec)

// COMMAND ----------

normDensND(Vector(0,0))

// COMMAND ----------

val mdeNodesToRec = mdeHist.counts.leaves.map(leaf => leaf -> tree.cellAt(leaf))

// COMMAND ----------

mdeNodesToRec.head._2

// COMMAND ----------

corners(mdeNodesToRec.head._2).map(normDensND(_))

// COMMAND ----------

import org.apache.commons.math3.distribution.NormalDistribution
val normal = new NormalDistribution()
val mdeNodesToNormalMuApprox = mdeNodesToRec.map{ case (node, rec) =>
//   val recCorners = corners(rec)
//   node -> recCorners.map(normDensND(_)).sum / recCorners.length
  node -> (normal.cumulativeProbability(rec.high.head) - normal.cumulativeProbability(rec.low.head))
}

// COMMAND ----------

mdeNodesToNormalMuApprox.unzip._1 == toDensityHistogram(mdeHist).densityMap.leaves

// COMMAND ----------

toDensityHistogram(mdeHist).densityMap.vals.unzip._1

// COMMAND ----------

val l1errorApprox = toDensityHistogram(mdeHist).densityMap.vals.map{ case (dens, vol) => dens * vol }.zip(mdeNodesToNormalMuApprox.unzip._2).map{ case (dhat, d) => abs(dhat - d)}

// COMMAND ----------

l1errorApprox.sum
