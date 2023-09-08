// Databricks notebook source
// MAGIC %md
// MAGIC ## 1TB 10-dimensional Cross of Gaussians
// MAGIC
// MAGIC This project was partly supported by Combient Mix AB through 2023 summer internship in Data Engineering Sciences to Axel Sandstedt and a grant from Wallenberg AI, Autonomous Systems and Software Program funded by Knut and Alice Wallenberg Foundation to Raazesh Sainudiin.
// MAGIC
// MAGIC This notebook was used to time a 1TB dataset estimation using the notebook's 10 dimensional cross mixture distribution. The cluster configuration used can be seen below.
// MAGIC
// MAGIC
// MAGIC #### Packages used:
// MAGIC #####Needed for RNG/sampling inside library
// MAGIC - commons_rng_client_api_1_5.jar
// MAGIC - commons_rng_core_1_5.jar
// MAGIC - commons_rng_sampling_1_5.jar
// MAGIC - commons_rng_simple_1_5.jar
// MAGIC  
// MAGIC #####Needed for Mixture Distribution
// MAGIC - commons_numbers_core_1_1.jar
// MAGIC - commons_numbers_gamma_1_1.jar
// MAGIC - commons_statistics_distribution_1_0.jar
// MAGIC
// MAGIC #####SparkDensityTree
// MAGIC - disthist_2_12_0_1_0.jar

// COMMAND ----------

// MAGIC %md
// MAGIC {\
// MAGIC     "num_workers": 16,\
// MAGIC     "cluster_name": "SparkDensityTree-16-Workers",\
// MAGIC     "spark_version": "12.2.x-scala2.12",\
// MAGIC     "spark_conf": {},\
// MAGIC     "aws_attributes": {\
// MAGIC         "first_on_demand": 1,\
// MAGIC         "availability": "SPOT_WITH_FALLBACK",\
// MAGIC         "zone_id": "eu-west-1c",\
// MAGIC         "spot_bid_price_percent": 100,\
// MAGIC         "ebs_volume_type": "GENERAL_PURPOSE_SSD",\
// MAGIC         "ebs_volume_count": 3,\
// MAGIC         "ebs_volume_size": 100\
// MAGIC     },\
// MAGIC     "node_type_id": "c4.4xlarge",\
// MAGIC     "driver_node_type_id": "c4.4xlarge",\
// MAGIC     "ssh_public_keys": [],\
// MAGIC     "custom_tags": {},\
// MAGIC     "spark_env_vars": {\
// MAGIC         "PYSPARK_PYTHON": "/databricks/python3/bin/python3"\
// MAGIC     },\
// MAGIC     "autotermination_minutes": 10,\
// MAGIC     "enable_elastic_disk": false,\
// MAGIC     "cluster_source": "UI",\
// MAGIC     "init_scripts": [],\
// MAGIC     "single_user_name": "sandstedt225@gmail.com",\
// MAGIC     "enable_local_disk_encryption": false,\
// MAGIC     "data_security_mode": "SINGLE_USER",\
// MAGIC     "runtime_engine": "STANDARD",\
// MAGIC     "cluster_id": "0902-172105-9v20sz5m"\
// MAGIC }

// COMMAND ----------

import org.apache.spark._
import org.apache.spark.mllib.linalg.{ Vector => MLVector, _ }
import org.apache.spark.mllib.random.RandomRDDs.normalVectorRDD
import org.apache.spark.sql.{ Dataset, SparkSession }
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import scala.math.BigInt
import java.math.BigInteger

import org.apache.commons.rng.simple.RandomSource
import org.apache.commons.rng.UniformRandomProvider
import org.apache.commons.rng.sampling.distribution.SharedStateDiscreteSampler
import org.apache.commons.rng.sampling.distribution.AliasMethodDiscreteSampler
import org.apache.commons.statistics.distribution.BetaDistribution

import scala.math.{min, max, abs, sqrt, sin, cos}
import scala.collection.immutable.{Vector => Vec}

import co.wiklund.disthist._
import co.wiklund.disthist.Types._
import co.wiklund.disthist.RectangleFunctions._
import co.wiklund.disthist.MDEFunctions._
import co.wiklund.disthist.LeafMapFunctions._
import co.wiklund.disthist.SpatialTreeFunctions._
import co.wiklund.disthist.HistogramFunctions._
import co.wiklund.disthist.TruncationFunctions._
import co.wiklund.disthist.MergeEstimatorFunctions._
import co.wiklund.disthist.SubtreePartitionerFunctions._

// COMMAND ----------

/**
 * LineMixture - Describes the mixtures of normal distributions equidistant on a line, beginning at start and ending at end.
 *               The weight each mixture is given is determined using a beta distribution heuristic; The beta distribution according to the given
 *               alpha and beta parameters are "stretched out" along the line and every mixture's weight is determined by the distribution's density
 *               at that point. The weights are then normalized to have a total sum of 1 which gives the wanted probabilities.
 *
 * @param start - Starting point of mixture line
 * @param end - Ending point of mixture line
 * @param numMixtures - Number of mixtures along line
 * @param alpha - alpha parameter in Beta distribution
 * @param beta - beta parameter in Beta distribution
 * @param scales - Optional scaling of each mixture 
 */
case class LineMixture(start : Array[Double], end : Array[Double], numMixtures : Int, alpha : Double = 1.0, beta : Double = 1.0, scales : Array[Double] = Array(1.0)) extends Serializable {

  /* Calculate mixture probabilities and setup discrete sampling probabilities */

  val distance : Double = sqrt((end zip start).map(t => t._1 - t._2).map(x => x*x).reduce(_+_))
  val dir : Array[Double] = (end zip start).map(t => (t._1 - t._2) / distance)
  val equiDistance = distance / numMixtures

  var probabilities : Array[Double] = new Array(numMixtures)
  var mixtureMeans : Array[Array[Double]] = Array.ofDim[Double](numMixtures, start.length)

  /* Scope this so b becomes a temporary variable, it is not serializable so must be done */
  {
    val b : BetaDistribution = BetaDistribution.of(alpha, beta)
    for (i <- 0 until numMixtures) {
      val lineTranslation = dir.map(_ * (0.5 + i) * equiDistance)
      mixtureMeans(i) = (start zip lineTranslation).map(t => t._1 + t._2)
      probabilities(i) = b.density((0.5 + i) / numMixtures)
    }
    probabilities = probabilities.map(w => w / probabilities.reduce(_+_))
  }

  /**
   * getMixtureSampler - index sampler for mixtures on line, may need to create several per class
   */
  def getMixtureSampler(rng : UniformRandomProvider) : SharedStateDiscreteSampler = {
    AliasMethodDiscreteSampler.of(rng, probabilities)
  }

  /**
   * transform - scale a standard gaussian sample and translate it to the given mean [ N(0,I) => N(mean, scale*I) ]
   */
  def transform(p : MLVector, mean : Array[Double], scale : Double) : MLVector = {
    val arr = p.toArray
    val res : Array[Double] = new Array(arr.length)
    for (i <- 0 until mean.length) {
      res(i) = arr(i) * scale + mean(i)
    }
    Vectors.dense(res)
  }

  /**
   * sample - sample a point from the mixtures on the line
   *
   * @param mixtureSampler - mixture of distributions on line
   * @param v - Standard Normal sample to transform to some mixture on line
   */
  def sample (mixtureSampler : SharedStateDiscreteSampler, p : MLVector) : MLVector = {
    val mIndex = mixtureSampler.sample()
    transform(p, mixtureMeans(mIndex), scales(mIndex % scales.length))
  }
}


// COMMAND ----------

/**
 * MixtureDistribution - Mixure of distributions along a set of lines, each with a given weight. The weights are then
 *    normalized to a total sum of 1 so that we work with probabilities.
 *
 *  @param lineMixtures - A set of lines with equidistant mixtures on them
 *  @param lineWeights - positive weights corresponding to non-normalized probabilities.
 */
case class MixtureDistribution(lineMixtures : Array[LineMixture], lineWeights : Array[Double]) extends Serializable {

  val probabilities = lineWeights.map(w => w / lineWeights.reduce(_+_))

  /**
   * sample - Takes an iterator of a sample from N(0,I), and transforms the sample to the given mixture distribution.
   */
  def sample(stdNormalSample : Iterator[MLVector]) : Iterator[MLVector] = {

    val rng : UniformRandomProvider = RandomSource.XO_RO_SHI_RO_128_PP.create()
    val lineSampler : SharedStateDiscreteSampler = AliasMethodDiscreteSampler.of(rng, probabilities)
    var mixturesOnLineSamplers : Array[SharedStateDiscreteSampler] = new Array(lineMixtures.length)

    for (i <- 0 until lineMixtures.length) {
      mixturesOnLineSamplers(i) = lineMixtures(i).getMixtureSampler(rng)
    }
    
    stdNormalSample.map(p => {
      val line = lineSampler.sample()
      lineMixtures(line).sample(mixturesOnLineSamplers(line), p)
    })
  }
}


// COMMAND ----------

val start1 = Array(1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0)
val end1 =   Array(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
val scale1 : Array[Double] = Array(0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4)
val line1 = LineMixture(start1, end1, 10, 0.25, 0.25, scale1)

val start2 = Array(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
val end2 =   Array(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
val scale2 : Array[Double] = Array(0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4)
val line2 = LineMixture(start2, end2, 10, 0.25, 0.25, scale2)

val mixture : MixtureDistribution = MixtureDistribution(Array(line1, line2), Array(1.0, 1.0))

// COMMAND ----------

// MAGIC %md
// MAGIC #### IMPORTANT
// MAGIC Set the **pathToDataFolder** to your folder of choice. If you later want to use the **visualisation** stuff in the second half of the notebook, you will have to manually set the python variable **path** and make it the same value as rootPath.

// COMMAND ----------

val mixtureName = "crossed_mixture"
val dimensions = 10

val pathToDataFolder = "dbfs:/Users/sandstedt225@gmail.com/data/1TB/MIXTURE_10D"
val rootPath = s"${pathToDataFolder}/${mixtureName}/${dimensions}/"

// COMMAND ----------

// MAGIC %python
// MAGIC path = "dbfs:/Users/sandstedt225@gmail.com/data/1TB/MIXTURE_10D/crossed_mixture/10//"

// COMMAND ----------

val trainSize : Long = (math.pow(10.toLong,11) * (7.0/80.0)).toLong
val validationSize : Long = (math.pow(10.toLong,11) * (3.0/80.0)).toLong

/* Used for checkpoint before the subtree merging stuff, useful when a failure at that stage becomes costly. */
val trainingPath = rootPath + "countedTrain"

/* Path to finest histogram we are willing to consider as an estimate (Used as checkpoint, Output of Subtree Merging step, Stage 3) */
val finestHistPath = rootPath + "finestHist"

/* Path to final histogram estimate */ 
val mdeHistPath = rootPath + "mdeHist"

/* Path to saved root box */
val treePath = rootPath + "spatialTree"

/* Path to saved depth used for labelling, or finding the leaf addresses with data points in them */
val finestResDepthPath = rootPath + "finestRes"


// COMMAND ----------

// MAGIC %md
// MAGIC We set the number of partitions used to something appropriate.
// MAGIC Why do we need **numCores**? This will be used in the last stage in the MDE search, so it needs to be set.

// COMMAND ----------

val numCores = 256
val partsScale = 256
val numTrainingPartitions = 32*partsScale
val numValidationPartitions = 16*partsScale

// COMMAND ----------

// MAGIC %md
// MAGIC Next, we setup the mixture sample RDDs

// COMMAND ----------

val trainingSeed = 7885643
val validationSeed = 25394782
val trainingRDD = normalVectorRDD(spark.sparkContext, trainSize, dimensions, numTrainingPartitions, trainingSeed).mapPartitions(iter => mixture.sample(iter))
val validationRDD = normalVectorRDD(spark.sparkContext, validationSize, dimensions, numValidationPartitions, validationSeed).mapPartitions(iter => mixture.sample(iter))

// COMMAND ----------

// MAGIC %md
// MAGIC First up is deriving the box hull of the validation and training data which will be our **root regular raving**.

// COMMAND ----------

/* Get boxhull of training data and test data */
var rectTrain = RectangleFunctions.boundingBox(trainingRDD)
var rectValidation = RectangleFunctions.boundingBox(validationRDD)
val rootBox = RectangleFunctions.hull(rectTrain, rectValidation)

// COMMAND ----------

// MAGIC %md
// MAGIC Our choice of depth to split down to is the depth at which every leaf's cell has no side with a length larger than 1e-1. We save the stuff since we will need it to recreate our
// MAGIC histogram estimate later. `finestResDepth` will come in handy when we apply our regressions tools.

// COMMAND ----------

val finestResSideLength = 1e-1 
val tree = widestSideTreeRootedAt(rootBox)
val finestResDepth = tree.descendBoxPrime(Vectors.dense(rootBox.low.toArray)).dropWhile(_._2.widths.max > finestResSideLength).head._1.depth

Vector(tree.rootCell.low, tree.rootCell.high).toIterable.toSeq.toDS.write.mode("overwrite").parquet(treePath) 
Array(finestResDepth).toIterable.toSeq.toDS.write.mode("overwrite").parquet(finestResDepthPath) 

// COMMAND ----------

// MAGIC %md A quick check that everything is saved correctly. Due to the way we save stuff, they might now necessarily be loaded in in the same order, so we fix the rootbox sides.

// COMMAND ----------

  val treeVec = spark.read.parquet(treePath).as[Vec[Double]].collect
  val lowArr : Array[Double] = new Array(dimensions)
  val highArr : Array[Double] = new Array(dimensions)
  for (j <- 0 until dimensions) {
    if(treeVec(0)(j) < treeVec(1)(j)) {
      lowArr(j) = treeVec(0)(j)
      highArr(j) = treeVec(1)(j)
    } else {
      lowArr(j) = treeVec(1)(j)
      highArr(j) = treeVec(0)(j)
    }
  }

  val tree = widestSideTreeRootedAt(Rectangle(lowArr.toVector, highArr.toVector))
  val finestResDepth = spark.read.parquet(finestResDepthPath).as[Depth].collect()(0)
          

// COMMAND ----------

// MAGIC %md
// MAGIC Next up is to find the leaf box adress (its path starting from the root), label, for every leaf with a data point inside of it. A reduceByKey shuffle is then applied (inside `quickToLabeled`) so that we only have unique leaves and their individual data point counts. You should probably **persist** this rdd in MEMORY_AND_DISK or save it to disk as we do here. You will usually apply two actions on the RDD, so you do not want to 
// MAGIC recompute this step twice, it contains an expensive shuffle. It is also the next section that is the most error-prone part of the library, so having a checkpoint here is nice.
// MAGIC  Furthermore, if you save it using parquet, the depth may not be larger than roughly 128. 
// MAGIC If it is, the underlying BigInteger will lose bits (due to some wonky sql reason?), so do not do it.
// MAGIC If the depth is larger than the given 
// MAGIC number, the scala code 4 boxes below shows how to do it correctly. This is more costly so we did not do it. 

// COMMAND ----------

val countedTrain = quickToLabeled(tree, finestResDepth, trainingRDD)
/* Only works for depth < 128 */
dbutils.fs.rm(trainingPath, true)
countedTrain.toDS.write.mode("overwrite").parquet(trainingPath)
trainingRDD.unpersist()

// COMMAND ----------

// MAGIC %md
// MAGIC Here we set a count limit. We set a minimum limit of 20000, but if we find a leaf with a maximum leaf count larger than the minimum, we pick that one instead. If you do not want to print
// MAGIC the maxLeafCount, just use the commented line method `getCountLimit`. It does the same thing as the code being used.

// COMMAND ----------

val minimumCountLimit = 20000
val countedTrain = spark.read.parquet(trainingPath).as[(NodeLabel, Count)].rdd
val maxLeafCount = countedTrain.map(_._2).reduce(max(_,_))
println("Max is count is " + maxLeafCount + " at depth " + finestResDepth)
val countLimit = max(minimumCountLimit, maxLeafCount)

//val countLimit = getCountLimit(countedTrain, minimumCountLimit)

// COMMAND ----------

// MAGIC %md
// MAGIC It is time to take our leaf data at the finest resolution and merge the leaves up to the count limit which produces the most refined histogram we are willing to take as a density estimate.
// MAGIC `sampleSizeHint` it roughly the number of points we sample to the driver for every new partition. The driver estimates the probability distribution among the branches of the tree, after which
// MAGIC those branches are assigned to new partitions.

// COMMAND ----------

implicit val ordering : Ordering[NodeLabel] = leftRightOrd
val sampleSizeHint = 1000
val partitioner = new SubtreePartitioner(numTrainingPartitions, countedTrain, sampleSizeHint)
val depthLimit = partitioner.maxSubtreeDepth
val subtreeRDD = countedTrain.repartitionAndSortWithinPartitions(partitioner)
val finestHistogram : Histogram = mergeLeavesHistogram(tree, subtreeRDD, countLimit, depthLimit)

// COMMAND ----------

// MAGIC %md
// MAGIC Here you can see how to safely save our (NodeLabel, Count)'s to disk; You'll have to use this if the depth of the leaves are > 126 or so. We do this as a checkpoint before the last stage. 

// COMMAND ----------

finestHistogram.counts.toIterable.toSeq.map(t => (t._1.lab.bigInteger.toByteArray, t._2)).toDS.write.mode("overwrite").parquet(finestHistPath)

// COMMAND ----------

// MAGIC %md
// MAGIC We reload the histogram

// COMMAND ----------

val counts = spark.read.parquet(finestHistPath).as[(Array[Byte], Count)].map(t => (NodeLabel(new BigInt(new BigInteger(t._1))), t._2)).collect
val finestHistogram = Histogram(tree, counts.map(_._2).reduce(_+_), fromNodeLabelMap(counts.toMap)), 

// COMMAND ----------



// COMMAND ----------

val countedValidation = quickToLabeledNoReduce(tree, finestResDepth, validationRDD)
//spark.conf.set("spark.default.parallelism", s"${numValidationPartitions}")

// COMMAND ----------

// MAGIC %md
// MAGIC Now we generate an estimate. We give the method the finest histogram, our validation data and the number of validation points. `kInMDE` determines how many histograms we will consider
// MAGIC in every iterations, I've always used a value of 10 here, but depending on the situation, you may use many more probably. The `numCores` should have been set to the number of cores in the cluster.

// COMMAND ----------

val kInMDE = 10
val mdeHist = getMDE(
  finestHistogram,
  countedValidation, 
  validationSize,
  kInMDE, 
  numCores,
  true 
)

// COMMAND ----------

// MAGIC %md
// MAGIC We save the histogram estimate to disk. Make sure you have saved the tree structure and the finestDepthResolution at the start, they will be needed if you wish to recreate the histogram
// MAGIC and to certain regression stuff with it.

// COMMAND ----------

mdeHist.counts.toIterable.toSeq.map(t => (t._1.lab.bigInteger.toByteArray, t._2)).toDS.write.mode("overwrite").parquet(mdeHistPath)
val density = toDensityHistogram(mdeHist).normalize

// COMMAND ----------

val treeVec = spark.read.parquet(treePath).as[Vec[Double]].collect
val lowArr : Array[Double] = new Array(dimensions)
val highArr : Array[Double] = new Array(dimensions)
for (j <- 0 until dimensions) {
  if(treeVec(0)(j) < treeVec(1)(j)) {
    lowArr(j) = treeVec(0)(j)
    highArr(j) = treeVec(1)(j)
  } else {
    lowArr(j) = treeVec(1)(j)
    highArr(j) = treeVec(0)(j)
  }
}

val tree = widestSideTreeRootedAt(Rectangle(lowArr.toVector, highArr.toVector))
val finestResDepth = spark.read.parquet(finestResDepthPath).as[Depth].collect()(0)
val mdeHist = {
  val mdeCounts = spark.read.parquet(mdeHistPath).as[(Array[Byte], Count)].map(t => (NodeLabel(BigInt(new BigInteger(t._1))), t._2))
  Histogram(tree, mdeCounts.map(_._2).reduce(_+_), fromNodeLabelMap(mdeCounts.collect.toMap))
}
val density = toDensityHistogram(mdeHist).normalize

// COMMAND ----------

def saveSample(density : DensityHistogram, sampleSize : Int, dimensions : Int, limitsPath : String, samplePath : String, seed : Long) = {

  val limits : Array[Double] = Array(
    density.tree.rootCell.low(0),
    density.tree.rootCell.high(0),
    density.tree.rootCell.low(1),
    density.tree.rootCell.high(1),
  )
  Array(limits).toIterable.toSeq.toDS.write.mode("overwrite").parquet(limitsPath)

  val rng : UniformRandomProvider = RandomSource.XO_RO_SHI_RO_128_PP.create(seed)
  val sample = density.sample(rng, sampleSize).map(_.toArray)

  var arr : Array[Double] = new Array(dimensions * sample.length)
  for (i <- 0 until sample.length) {
      for (j <- 0 until dimensions) {
        arr(j + dimensions*i) = sample(i)(j)
      }
  }
 
  Array(arr).toIterable.toSeq.toDS.write.mode("overwrite").parquet(samplePath)
}    

// COMMAND ----------

// MAGIC %python
// MAGIC from matplotlib import cbook
// MAGIC from matplotlib import cm
// MAGIC from matplotlib.colors import LightSource
// MAGIC import matplotlib.pyplot as plt
// MAGIC from matplotlib.ticker import LinearLocator
// MAGIC import matplotlib_inline.backend_inline
// MAGIC from matplotlib.patches import Rectangle, PathPatch
// MAGIC import mpl_toolkits.mplot3d.art3d as art3d
// MAGIC from matplotlib.transforms import Bbox
// MAGIC import numpy as np
// MAGIC
// MAGIC def scatterPlot(dimensions, alph, limitsPath, samplePath):
// MAGIC
// MAGIC     matplotlib_inline.backend_inline.set_matplotlib_formats('png2x')
// MAGIC
// MAGIC     limits = np.array(spark.read.parquet(limitsPath).collect())[-1,-1]
// MAGIC     values = np.array(spark.read.parquet(samplePath).collect())[-1,-1]
// MAGIC
// MAGIC     #fig, ax = plt.subplots()
// MAGIC
// MAGIC     fig, axs = plt.subplots(2, 2)
// MAGIC
// MAGIC     length = int(len(values) / 3)
// MAGIC
// MAGIC     xs = np.ndarray(shape=(dimensions, length))
// MAGIC     for i in range(dimensions):
// MAGIC         for j in range(length):
// MAGIC             xs[i,j] = values[3*j + i]
// MAGIC
// MAGIC     axs[0,0].scatter(xs[0,], xs[1,], alpha = alph)
// MAGIC     axs[0, 0].set_title('X2,X4')
// MAGIC     #axs[0,0].set_xlabel('X2')
// MAGIC     #axs[0,0].set_ylabel('X4')
// MAGIC     axs[0,1].scatter(xs[0,], xs[2,], alpha = alph)
// MAGIC     axs[0, 1].set_title('X2,X5')
// MAGIC     #axs[0,1].set_xlabel('X2')
// MAGIC     #axs[0,1].set_ylabel('X5')
// MAGIC     axs[1,1].scatter(xs[1,], xs[2,], alpha = alph)
// MAGIC     axs[1, 1].set_xlabel('X4,X5')
// MAGIC     #axs[1,1].set_xlabel('X4')
// MAGIC     #axs[1,1].set_ylabel('X5')
// MAGIC
// MAGIC     # Customize the z axis.
// MAGIC     #ax.set_xlim(x4_min, x4_max)
// MAGIC     #ax.set_ylim(x6_min, x6_max)
// MAGIC     #ax.set_xlabel('X1')
// MAGIC     #ax.set_ylabel('X2')
// MAGIC
// MAGIC     plt.show()

// COMMAND ----------

val seed : Long = 123463
val limitsPath = s"${rootPath}/limits"
val samplePath = s"${rootPath}/sample"

val splitOrder = tree.splitOrderToDepth(finestResDepth)
val conditional = quickSlice(density, Vector(0,2,5,6,7,8,9), Vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), splitOrder).normalize

saveSample(conditional, 200, 3, limitsPath, samplePath, seed)

// COMMAND ----------

// MAGIC %python
// MAGIC scatterPlot(3, 1, path + "limits", path + "sample")

// COMMAND ----------

val seed : Long = 123463
val limitsPath = s"${rootPath}/limits"
val samplePath = s"${rootPath}/sample"

val splitOrder = tree.splitOrderToDepth(finestResDepth)
val conditional = quickSlice(density, Vector(0,2,5,6,7,8,9), Vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), splitOrder).normalize

saveSample(conditional, 5000, 3, limitsPath, samplePath, seed)

// COMMAND ----------

// MAGIC %python
// MAGIC scatterPlot(3, 0.02, path + "limits", path + "sample")

// COMMAND ----------

val seed : Long = 853632
val limitsPath = s"${rootPath}/limits"
val samplePath = s"${rootPath}/sample"

val splitOrder = tree.splitOrderToDepth(finestResDepth)
val conditional = quickSlice(density, Vector(0,2,5,6,7,8,9), Vector(0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75), splitOrder).normalize

saveSample(conditional, 200, 3, limitsPath, samplePath, seed)

// COMMAND ----------

// MAGIC %python
// MAGIC scatterPlot(3, 1, path + "limits", path + "sample")

// COMMAND ----------

val seed : Long = 853632
val limitsPath = s"${rootPath}/limits"
val samplePath = s"${rootPath}/sample"

val splitOrder = tree.splitOrderToDepth(finestResDepth)
val conditional = quickSlice(density, Vector(0,2,5,6,7,8,9), Vector(0.825, 0.825, 0.825, 0.825, 0.825, 0.825, 0.825), splitOrder).normalize

saveSample(conditional, 5000, 3, limitsPath, samplePath, seed)

// COMMAND ----------

// MAGIC %python
// MAGIC scatterPlot(3, 0.02, path + "limits", path + "sample")
