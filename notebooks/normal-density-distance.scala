// Databricks notebook source
import co.wiklund.disthist._
import SpatialTreeFunctions._
import MergeEstimatorFunctions._
import HistogramFunctions._

import LeafMapFunctions._
import MDEFunctions._
import Types._

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.random.RandomRDDs.normalVectorRDD


import spark.implicits._

// COMMAND ----------

val numPartitions = 16
spark.conf.set("spark.default.parallelism", numPartitions.toString)

val dimensions = 5
val sizeExp = 7
val deltas = Vector(0,1,2,3,5)

def getPath(delta: Int): String = s"dbfs:/Users/johannes.graner@gmail.com/sparkDensityTree/normal/${dimensions}d_${sizeExp}n/diff/${delta}delta/"

// COMMAND ----------

if(false) {
  for (delta <- deltas) {

    val trainSize = math.pow(10, sizeExp).toLong / 2
    val testSize = trainSize / 4
    val finestResSideLength = 1e-5

    val rootPath = s"dbfs:/Users/johannes.graner@gmail.com/sparkDensityTree/normal/${dimensions}d_${sizeExp}n/"
    val diffPath = rootPath + s"diff/${delta}delta/"
    val trainingPath = diffPath + "train"
    val labeledTrainPath = diffPath + "labeledTrain"
    val testPath = diffPath + "test"
    val labeledTestPath = diffPath + "labeledTest"

    val histogramPath = diffPath + "trainedHistogram"
    val mdeHistPath = diffPath + "mdeHist"

    val trainingPath0 = rootPath + "diff/0delta/train"
    val testPath0 = rootPath + "diff/0delta/test"

    val rootBox = Rectangle(
      (1 to dimensions).toVector.map(_ => -10.0),
      (1 to dimensions).toVector.map(_ => 15.0)
    )
    val tree = widestSideTreeRootedAt(rootBox)
    val finestResDepth = tree.descendBoxPrime(Vectors.dense(rootBox.low.toArray)).dropWhile(_._2.widths.max > finestResSideLength).head._1.depth

    val countLimit = 400
    val stepSize = math.ceil(finestResDepth / 8.0).toInt
    val kInMDE = 10

    val offset: Double = delta / math.sqrt(dimensions)

    val rawTrainRDD = { d: Double => normalVectorRDD(spark.sparkContext, trainSize, dimensions, numPartitions * dimensions, 1234 + delta).map(vec => Vectors.dense(vec.toArray.map(x => x + offset))) }.apply(delta)
    val labeledTrainRDD = labelAtDepth(tree, finestResDepth, rawTrainRDD)
    spark.createDataset(labeledTrainRDD.map{ case (lab, vec) => (NodeLabel(lab), vec.toArray)}).write.mode("overwrite").parquet(labeledTrainPath)
    val labeledTrainDS = spark.read.parquet(labeledTrainPath).as[(NodeLabel, Array[Double])]
    labeledToCountedDS(labeledTrainDS).write.mode("overwrite").parquet(trainingPath)

    val rawTestRDD = { d: Double => normalVectorRDD(spark.sparkContext, testSize, dimensions, numPartitions * dimensions, 4321 - delta).map(vec => Vectors.dense(vec.toArray.map(x => x + offset))) }.apply(delta)
    val labeledTestRDD = labelAtDepth(tree, finestResDepth, rawTestRDD)
    spark.createDataset(labeledTestRDD.map{ case (lab, vec) => (NodeLabel(lab), vec.toArray)}).write.mode("overwrite").parquet(labeledTestPath)
    val labeledTestDS = spark.read.parquet(labeledTestPath).as[(NodeLabel, Array[Double])]
    labeledToCountedDS(labeledTestDS).write.mode("overwrite").parquet(testPath)

    val countedTrainDS = spark.read.parquet(trainingPath).as[(NodeLabel, Count)]
    val countedTestDS = spark.read.parquet(testPath).as[(NodeLabel, Count)].cache

    dbutils.fs.rm(histogramPath, true)
    val mergedDS = mergeLeaves(tree, countedTrainDS, countLimit, stepSize, histogramPath, true)

    val mde = getMDE(
      Histogram(tree, mergedDS.map(_._2).reduce(_+_), fromNodeLabelMap(mergedDS.collect.toMap)), 
      countedTestDS, 
      kInMDE, 
      true
    )
    mde.counts.toIterable.toSeq.toDS.write.mode("overwrite").parquet(mdeHistPath)

  }
}

// COMMAND ----------

  val rootBox = Rectangle(
    (1 to dimensions).toVector.map(_ => -10.0),
    (1 to dimensions).toVector.map(_ => 15.0)
  )
  val tree = widestSideTreeRootedAt(rootBox)

// COMMAND ----------

val mdeHists = deltas.map{ delta =>
  val mdeCounts = spark.read.parquet(getPath(delta) + "mdeHist").as[(NodeLabel, Count)]
  Histogram(tree, mdeCounts.map(_._2).reduce(_+_), fromNodeLabelMap(mdeCounts.collect.toMap))
}

// COMMAND ----------

if (dimensions == 2)
  mdeHists.zip(deltas).map{ case (mde, delta) => mde.density(Vectors.dense(delta / math.sqrt(dimensions), delta / math.sqrt(dimensions)))}
else
  mdeHists.zip(deltas).map{ case (mde, delta) => mde.density(Vectors.dense(delta / math.sqrt(dimensions), delta / math.sqrt(dimensions), delta / math.sqrt(dimensions), delta / math.sqrt(dimensions), delta / math.sqrt(dimensions)))}

// COMMAND ----------

val testRDDs = deltas.map{ delta =>
  val offset = delta / math.sqrt(dimensions)
  normalVectorRDD(spark.sparkContext, 10000, dimensions, numPartitions * dimensions, 4567 + delta).map(vec => Vectors.dense(vec.toArray.map(x => x + offset)))
}

// COMMAND ----------

val delta0RDD = testRDDs.head
val deltaRDDs = testRDDs.tail

// COMMAND ----------

val delta0Local = delta0RDD.collect
val deltaLocals = deltaRDDs.map(_.collect)

// COMMAND ----------

val test0deltaConfusion = deltas.tail.zip(mdeHists.tail).map{ case (delta, mde) =>
  val denses = delta0Local.map(x => (mdeHists.head.density(x), mde.density(x))).filterNot(d => d._1 == d._2)
  Map(0 -> denses.filter(d => d._1 > d._2).length, delta -> denses.filter(d => d._1 < d._2).length)
}

// COMMAND ----------

val testdeltaConfusions = deltaLocals.zip(deltas.tail).zip(mdeHists.tail).map{ case ((localrdd, delta), mde) =>
  val denses = localrdd.map(x => (mdeHists.head.density(x), mde.density(x))).filterNot(d => d._1 == d._2)
  Map(0 -> denses.filter(d => d._1 > d._2).length, delta -> denses.filter(d => d._1 < d._2).length)
}

// COMMAND ----------

case class confMatPerc(TP: Double, FP: Double, FN: Double, TN: Double)

case class confMat(TP: Int, FP: Int, FN: Int, TN: Int) {
  def total = TP + FP + FN + TN
  def toPerc = confMatPerc(TP.toDouble / total, FP.toDouble / total, FN.toDouble / total, TN.toDouble / total)
}

// COMMAND ----------

val confMatrices = test0deltaConfusion.zip(testdeltaConfusions).map{ case (p, n) => 
  val delta = (p.keySet diff Set(0)).toSeq.head
  confMat(p(0), p(delta), n(0), n(delta))
}

// COMMAND ----------

deltas.tail.zip(confMatrices.map(_.toPerc)).foreach{ case (delta, conf) => 
  println(s"delta: ${delta}")
  println(s"\t${conf.TP}\t${conf.FP}")
  println(s"\t${conf.FN}\t${conf.TN}")
}

// COMMAND ----------



// COMMAND ----------

display({
  val op: (Count, Count) => Count = _ + _
  val totalCount = mdeHists.head.counts.leaves.length + mdeHists.last.counts.leaves.length
  val hist = mdeHists.last //Histogram(tree, totalCount, mrpOperate(mdeHists.head.counts, mdeHists.last.counts, op, 0))
  val margHist = marginalize(hist, Vector(0))
  margHist.densityMap.toIterable.toVector.map{ case (node, (dens, _)) =>
    (margHist.tree.cellAt(node).centre(0), dens)
  }
}.toDF("x", "y"))
