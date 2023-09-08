// Databricks notebook source
// MAGIC %md
// MAGIC
// MAGIC ## 1TB 2-dimensional Cross of Gaussians
// MAGIC
// MAGIC This project was partly supported by Combient Mix AB through 2023 summer internship in Data Engineering Sciences to Axel Sandstedt and a grant from Wallenberg AI, Autonomous Systems and Software Program funded by Knut and Alice Wallenberg Foundation to Raazesh Sainudiin.
// MAGIC
// MAGIC This notebook was used time to a 1TB dataset estimation using the notebook's 2 dimensional cross mixture distribution. The cluster configuration used can be seen below.
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
 * MixtureDistribution - Mixture of distributions along a set of lines, each with a given weight. The weights are then
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

val numMixtures : Int = 10
val alpha : Double = 0.25
val beta : Double = 0.25

val start1 = Array(-5.0, -5.0)
val end1 = Array(5.0, 5.0)
val line1 = LineMixture(start1, end1, numMixtures, alpha, beta)

val start2 = Array(-5.0, 5.0)
val end2 = Array(5.0, -5.0)
val line2 = LineMixture(start2, end2, numMixtures, alpha, beta)

val lineWeights = Array(1.0, 1.0)
val mixture : MixtureDistribution = MixtureDistribution(Array(line1, line2), lineWeights)

// COMMAND ----------

// MAGIC %md
// MAGIC #### IMPORTANT
// MAGIC Set the **pathToDataFolder** to your folder of choice. If you later want to use the **visualisation** stuff in the second half of the notebook, you will have to manually set the python variable **path** and make it the same value as rootPath.

// COMMAND ----------

val mixtureName = "crossed_mixture"
val dimensions = 2

val pathToDataFolder = "dbfs:/Users/sandstedt225@gmail.com/data/1TB/MIXTURE_2D"
val rootPath = s"${pathToDataFolder}/${mixtureName}/${dimensions}/"

// COMMAND ----------

// MAGIC %python
// MAGIC path = "dbfs:/Users/sandstedt225@gmail.com/data/1TB/MIXTURE_2D/crossed_mixture/2/"

// COMMAND ----------

val trainSize : Long = (math.pow(10.toLong,11) * (7.0/16.0)).toLong
val validationSize : Long = (math.pow(10.toLong,11) * (3.0/16.0)).toLong

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
// MAGIC If we have 256 cores to distribute tasks between, and for 1TB data, this gives us roughly 1TB/16384 ~~ 100MB per partition, larger partitions than that for this
// MAGIC cluster configuration (16 Workers, c4.4xlarge) and we might crash on stage 3 and 4. Why do we need **numCores**? This will be used in the last stage in the MDE search, so it needs to be set.

// COMMAND ----------

val numCores = 256
val partsScale = 512
val numTrainingPartitions = 32*partsScale
val numValidationPartitions = 32*partsScale

// COMMAND ----------

// MAGIC %md
// MAGIC Next, we setup the mixture sample RDDs

// COMMAND ----------

val trainingSeed = 2253643
val validationSeed = 5534718
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
// MAGIC Our choice of depth to split down to is the depth at which every leaf's cell has no side with a length larger than 1e-5. We save the stuff since we will need it to recreate our
// MAGIC histogram estimate later. `finestResDepth` will come in handy when we apply our regressions tools.

// COMMAND ----------

val finestResSideLength = 1e-5 
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

// COMMAND ----------

// MAGIC %md
// MAGIC Here we set a count limit. We set a minimum limit of 100000, but if we find a leaf with a maximum leaf count larger than the minimum, we pick that one instead. If you do not want to print
// MAGIC the maxLeafCount, just use the commented line method `getCountLimit`. It does the same thing as the code being used.

// COMMAND ----------

val minimumCountLimit = 100000
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

// MAGIC %md
// MAGIC We must now label the validation data (deriving the leaf address of every data point). No reduceByKey is needed here, so use `quickToLabeledNoReduce`.

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

// MAGIC %md
// MAGIC #### Visualisation of densities with 2D support
// MAGIC It is all a big hack; don't expect stuff to work very well, use at risk of your own sanity. If you choose to try this stuff, make sure you set the python **path** variable correctly at the start of the notebook.

// COMMAND ----------

val treeVec = spark.read.parquet(treePath).as[Vec[Double]].collect
val lowArr : Array[Double] = new Array(2)
val highArr : Array[Double] = new Array(2)
for (j <- 0 to 1) {
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

def savePlotValues(density : DensityHistogram, rootCell : Rectangle, pointsPerAxis : Int, limitsPath : String, plotValuesPath : String) = {

  val limits : Array[Double] = Array(
    rootCell.low(0),
    rootCell.high(0),
    rootCell.low(1),
     rootCell.high(1),
    )
  Array(limits).toIterable.toSeq.toDS.write.mode("overwrite").parquet(limitsPath)

  val x4Width = rootCell.high(0) - rootCell.low(0)
  val x6Width = rootCell.high(1) - rootCell.low(1)

  val values : Array[Double] = new Array(pointsPerAxis * pointsPerAxis)

  for (i <- 0 until pointsPerAxis) {
    val x4_p = rootCell.low(0) + (i + 0.5) * (x4Width / pointsPerAxis)
    for (j <- 0 until pointsPerAxis) {
      val x6_p = rootCell.low(1) + (j + 0.5) * (x6Width / pointsPerAxis)
      values(i * pointsPerAxis + j) = density.density(Vectors.dense(x4_p, x6_p))
    }
  }
  Array(values).toIterable.toSeq.toDS.write.mode("overwrite").parquet(plotValuesPath)
}

// COMMAND ----------

def saveSupportPlot(density : DensityHistogram, rootCell : Rectangle, coverage : Double, limitsPath : String, supportPath : String) = {
  val coverageRegions : TailProbabilities = density.tailProbabilities
  
  val limits : Array[Double] = Array(
    rootCell.low(0),
    rootCell.high(0),
    rootCell.low(1),
     rootCell.high(1),
    )
  Array(limits).toIterable.toSeq.toDS.write.mode("overwrite").parquet(limitsPath)

  val x4Width = rootCell.high(0) - rootCell.low(0)
  val x6Width = rootCell.high(1) - rootCell.low(1)

  var n = 0
  /* bottom_left, width1, width2 */
  var values : Array[Double] = new Array(4 * density.densityMap.vals.length)
  for (i <- 0 until density.densityMap.vals.length) {
    val rect = tree.cellAt(density.densityMap.truncation.leaves(i))
    val centre = Vectors.dense(rect.centre(0), rect.centre(1))
      if (coverageRegions.query(centre) <= coverage) {
        values(n + 0) = rect.low(0)
        values(n + 1) = rect.low(1)
        values(n + 2) = rect.high(0) - rect.low(0)
        values(n + 3) = rect.high(1) - rect.low(1)
        n += 4
      }
  }

  Array(values.take(n)).toIterable.toSeq.toDS.write.mode("overwrite").parquet(supportPath)
}

// COMMAND ----------

def savePlotValuesCoverage(density : DensityHistogram, rootCell : Rectangle, coverage : Double, pointsPerAxis : Int, limitsPath : String, plotValuesPath : String) = {
  val coverageRegions : TailProbabilities = density.tailProbabilities
  
  val limits : Array[Double] = Array(
    rootCell.low(0),
    rootCell.high(0),
    rootCell.low(1),
     rootCell.high(1),
    )
  Array(limits).toIterable.toSeq.toDS.write.mode("overwrite").parquet(limitsPath)

  val x4Width = rootCell.high(0) - rootCell.low(0)
  val x6Width = rootCell.high(1) - rootCell.low(1)

  val values : Array[Double] = new Array(pointsPerAxis * pointsPerAxis)

  for (i <- 0 until pointsPerAxis) {
    val x4_p = rootCell.low(0) + (i + 0.5) * (x4Width / pointsPerAxis)
    for (j <- 0 until pointsPerAxis) {
      val x6_p = rootCell.low(1) + (j + 0.5) * (x6Width / pointsPerAxis)
      if (coverageRegions.query(Vectors.dense(x4_p, x6_p)) <= coverage)
        values(i * pointsPerAxis + j) = density.density(Vectors.dense(x4_p, x6_p))
      else
      values(i * pointsPerAxis + j) = 0.0
    }
  }
  Array(values).toIterable.toSeq.toDS.write.mode("overwrite").parquet(plotValuesPath)
}

// COMMAND ----------

// MAGIC %python
// MAGIC from matplotlib import cbook
// MAGIC from matplotlib import cm
// MAGIC from matplotlib.colors import LightSource
// MAGIC import matplotlib.pyplot as plt
// MAGIC from matplotlib.ticker import LinearLocator
// MAGIC import matplotlib_inline.backend_inline
// MAGIC import numpy as np
// MAGIC
// MAGIC def plotDensity(pointsPerAxis, z_max, limitsPath, valuesPath):
// MAGIC
// MAGIC     matplotlib_inline.backend_inline.set_matplotlib_formats('png2x')
// MAGIC
// MAGIC     limits = np.array(spark.read.parquet(limitsPath).collect())[-1,-1]
// MAGIC     values = np.array(spark.read.parquet(valuesPath).collect())[-1,-1]
// MAGIC
// MAGIC     x4_min = limits[0]
// MAGIC     x4_max = limits[1]
// MAGIC     x6_min = limits[2]
// MAGIC     x6_max = limits[3]
// MAGIC
// MAGIC     x4_width = (x4_max - x4_min) / pointsPerAxis
// MAGIC     x6_width = (x6_max - x6_min) / pointsPerAxis
// MAGIC
// MAGIC     x = np.arange(x4_min, x4_max, x4_width)
// MAGIC     y = np.arange(x6_min, x6_max, x6_width)
// MAGIC     x, y = np.meshgrid(x, y, indexing='ij')
// MAGIC
// MAGIC     z = np.empty((pointsPerAxis,pointsPerAxis))
// MAGIC     for i in range(pointsPerAxis):
// MAGIC         for j in range(pointsPerAxis):
// MAGIC             z[i,j] = values[i*pointsPerAxis + j]
// MAGIC
// MAGIC     fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
// MAGIC
// MAGIC     # Plot the surface.
// MAGIC     surf = ax.plot_surface(x, y, z, cmap=cm.gist_earth, linewidth=0, antialiased=False)
// MAGIC     # surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
// MAGIC     
// MAGIC     # Customize the z axis.
// MAGIC     ax.set_zlim(0.0, z_max)
// MAGIC     ax.set_xlim(x4_min, x4_max)
// MAGIC     ax.set_ylim(x6_min, x6_max)
// MAGIC     ax.set_xlabel('X1')
// MAGIC     ax.set_ylabel('X2')
// MAGIC     ax.set_zlabel('f_n(X1,X2)')
// MAGIC
// MAGIC     plt.show()

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
// MAGIC from matplotlib import colors
// MAGIC import numpy as np
// MAGIC
// MAGIC def plotDensityAndPlane(pointsPerAxis, z_max, limitsPath, valuesPath, planeAxis, planePoint):
// MAGIC
// MAGIC     matplotlib_inline.backend_inline.set_matplotlib_formats('png2x')
// MAGIC
// MAGIC     limits = np.array(spark.read.parquet(limitsPath).collect())[-1,-1]
// MAGIC     values = np.array(spark.read.parquet(valuesPath).collect())[-1,-1]
// MAGIC
// MAGIC     x4_min = limits[0]
// MAGIC     x4_max = limits[1]
// MAGIC     x6_min = limits[2]
// MAGIC     x6_max = limits[3]
// MAGIC
// MAGIC     x4_width = (x4_max - x4_min) / pointsPerAxis
// MAGIC     x6_width = (x6_max - x6_min) / pointsPerAxis
// MAGIC
// MAGIC     fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
// MAGIC
// MAGIC     zmax = 0
// MAGIC     for i in range(pointsPerAxis):
// MAGIC         for j in range(pointsPerAxis):
// MAGIC             if (values[i*pointsPerAxis + j] > zmax):
// MAGIC                 zmax = values[i*pointsPerAxis + j]
// MAGIC
// MAGIC     if (planeAxis == 0):
// MAGIC         pointBehindStart = np.arange(x4_min, planePoint, x4_width).size - 1
// MAGIC
// MAGIC         x1 = np.arange(x4_min, x4_min + x4_width * pointBehindStart, x4_width)
// MAGIC         y1 = np.arange(x6_min, x6_max, x6_width)
// MAGIC         xs1 = x1.size
// MAGIC         ys1 = y1.size
// MAGIC         x1, y1 = np.meshgrid(x1, y1, indexing='ij')
// MAGIC
// MAGIC         z1 = np.empty((xs1, ys1))
// MAGIC         for i in range(pointBehindStart):
// MAGIC             for j in range(pointsPerAxis):
// MAGIC                 z1[i, pointsPerAxis - 1 -j] = values[i*pointsPerAxis + j]
// MAGIC
// MAGIC         # Plot the surface.
// MAGIC         surf1 = ax.plot_surface(x1, y1, z1, cmap=cm.gist_earth, linewidth=0, antialiased=False, norm=colors.Normalize(vmin=0, vmax=zmax))
// MAGIC
// MAGIC         plane = Rectangle((x6_min, 0.0), x6_max - x6_min, z_max, alpha=.5, color="red")
// MAGIC         ax.add_patch(plane)
// MAGIC         art3d.pathpatch_2d_to_3d(plane, z=planePoint, zdir="x")
// MAGIC
// MAGIC         x2 = np.arange(planePoint, x4_max, x4_width)
// MAGIC         y2 = np.arange(x6_min, x6_max, x6_width)
// MAGIC         xs2 = x2.size
// MAGIC         ys2 = y2.size
// MAGIC         x2, y2 = np.meshgrid(x2, y2, indexing='ij')
// MAGIC
// MAGIC         z2 = np.empty((xs2, ys2))
// MAGIC         for i in range(pointBehindStart, pointsPerAxis):
// MAGIC             for j in range(pointsPerAxis):
// MAGIC                 z2[i - pointBehindStart, pointsPerAxis - 1 - j] = values[i*pointsPerAxis + j]
// MAGIC
// MAGIC         # Plot the surface.
// MAGIC         surf2 = ax.plot_surface(x2, y2, z2, cmap=cm.gist_earth, linewidth=0, antialiased=False, norm=colors.Normalize(vmin=0, vmax=zmax))
// MAGIC     else:
// MAGIC         pointBehindStart = np.arange(x6_min, planePoint, x6_width).size - 1
// MAGIC
// MAGIC         x1 = np.arange(x4_min, x4_max, x4_width)
// MAGIC         y1 = np.arange(planePoint, x6_max, x6_width)
// MAGIC         xs1 = x1.size
// MAGIC         ys1 = y1.size
// MAGIC         x1, y1 = np.meshgrid(x1, y1, indexing='ij')
// MAGIC         z1 = np.empty((xs1, ys1))
// MAGIC         for i in range(pointsPerAxis):
// MAGIC             for j in range(pointBehindStart, pointsPerAxis):
// MAGIC                 z1[i, j - pointBehindStart] = values[i*pointsPerAxis + j]
// MAGIC
// MAGIC         # Plot the surface.
// MAGIC         surf1 = ax.plot_surface(x1, y1, z1, cmap=cm.gist_earth, linewidth=0, antialiased=False, norm=colors.Normalize(vmin=0, vmax=zmax))
// MAGIC
// MAGIC         plane = Rectangle((x4_min, 0.0), x4_max - x4_min, z_max, alpha=.5, color="red")
// MAGIC         ax.add_patch(plane)
// MAGIC         art3d.pathpatch_2d_to_3d(plane, z=planePoint, zdir="y")
// MAGIC
// MAGIC         x2 = np.arange(x4_min, x4_max, x4_width)
// MAGIC         y2 = np.arange(x6_min, x6_min + pointBehindStart * x6_width, x6_width)
// MAGIC         x2, y2 = np.meshgrid(x2, y2, indexing='ij')
// MAGIC
// MAGIC         z2 = np.empty((pointsPerAxis, pointBehindStart))
// MAGIC         for i in range(pointsPerAxis):
// MAGIC             for j in range(pointBehindStart):
// MAGIC                 z2[i,j] = values[i*pointsPerAxis + j]
// MAGIC
// MAGIC         # Plot the surface.
// MAGIC         surf2 = ax.plot_surface(x2, y2, z2, cmap=cm.gist_earth, linewidth=0, antialiased=False, norm=colors.Normalize(vmin=0, vmax=zmax))
// MAGIC
// MAGIC     # Customize the z axis.
// MAGIC     ax.set_zlim(0.0, z_max)
// MAGIC     ax.set_xlim(x4_min, x4_max)
// MAGIC     ax.set_ylim(x6_min, x6_max)
// MAGIC     ax.set_xlabel('X1')
// MAGIC     ax.set_ylabel('X2')
// MAGIC     ax.set_zlabel('f_n(X1,X2)')
// MAGIC
// MAGIC     plt.show()

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
// MAGIC def plotConditionalDensity3D(pointsPerAxis, z_max, limitsPath, valuesPath, planeAxis, planePoint):
// MAGIC
// MAGIC     matplotlib_inline.backend_inline.set_matplotlib_formats('png2x')
// MAGIC
// MAGIC     limits = np.array(spark.read.parquet(limitsPath).collect())[-1,-1]
// MAGIC     values = np.array(spark.read.parquet(valuesPath).collect())[-1,-1]
// MAGIC
// MAGIC     x4_min = limits[0]
// MAGIC     x4_max = limits[1]
// MAGIC     x6_min = limits[2]
// MAGIC     x6_max = limits[3]
// MAGIC
// MAGIC     x4_width = (x4_max - x4_min) / pointsPerAxis
// MAGIC     x6_width = (x6_max - x6_min) / pointsPerAxis
// MAGIC
// MAGIC     fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
// MAGIC
// MAGIC     if (planeAxis == 0):
// MAGIC         pointBehindStart = np.arange(x4_min, planePoint, x4_width).size
// MAGIC         y1 = np.arange(x6_min, x6_max, x6_width)
// MAGIC         for i in range(pointsPerAxis):
// MAGIC             val = values[pointBehindStart * pointsPerAxis + (pointsPerAxis - 1) + i]
// MAGIC             plane = Rectangle((x6_min + i * x6_width, 0.0), x6_width, val, alpha=.15, color="blue")
// MAGIC             ax.add_patch(plane)
// MAGIC             art3d.pathpatch_2d_to_3d(plane, z=planePoint, zdir="x")
// MAGIC     else:
// MAGIC         pointBehindStart = np.arange(x6_min, planePoint, x6_width).size
// MAGIC         x1 = np.arange(x4_min, x4_max, x4_width)
// MAGIC         for i in range(pointsPerAxis):
// MAGIC             val = values[i * pointsPerAxis + pointBehindStart]
// MAGIC             plane = Rectangle((x4_min + i * x4_width, 0.0), x4_width, val, alpha=.15, color="blue")
// MAGIC             ax.add_patch(plane)
// MAGIC             art3d.pathpatch_2d_to_3d(plane, z=planePoint, zdir="y")
// MAGIC
// MAGIC     # Customize the z axis.
// MAGIC     ax.set_zlim(0.0, z_max)
// MAGIC     ax.set_xlim(x4_min, x4_max)
// MAGIC     ax.set_ylim(x6_min, x6_max)
// MAGIC     ax.set_xlabel('X1')
// MAGIC     ax.set_ylabel('X2')
// MAGIC     ax.set_zlabel('f_n(X1,X2)')
// MAGIC
// MAGIC     plt.show()

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
// MAGIC def supportPlot(z_max, limitsPath, supportPath):
// MAGIC
// MAGIC     matplotlib_inline.backend_inline.set_matplotlib_formats('png2x')
// MAGIC
// MAGIC     limits = np.array(spark.read.parquet(limitsPath).collect())[-1,-1]
// MAGIC     values = np.array(spark.read.parquet(valuesPath).collect())[-1,-1]
// MAGIC
// MAGIC     x4_min = limits[0]
// MAGIC     x4_max = limits[1]
// MAGIC     x6_min = limits[2]
// MAGIC     x6_max = limits[3]
// MAGIC
// MAGIC     x4_width = (x4_max - x4_min) / pointsPerAxis
// MAGIC     x6_width = (x6_max - x6_min) / pointsPerAxis
// MAGIC
// MAGIC     fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
// MAGIC
// MAGIC     for i in range(values.length() / 4):
// MAGIC         val = values[pointBehindStart * pointsPerAxis + (pointsPerAxis - 1) + i]
// MAGIC         plane = Rectangle((x6_min + i * x6_width, 0.0), x6_width, val, alpha=.15, color="blue")
// MAGIC         ax.add_patch(plane)
// MAGIC         art3d.pathpatch_2d_to_3d(plane, z=planePoint, zdir="x")
// MAGIC
// MAGIC
// MAGIC     # Customize the z axis.
// MAGIC     ax.set_zlim(0.0, z_max)
// MAGIC     ax.set_xlim(x4_min, x4_max)
// MAGIC     ax.set_ylim(x6_min, x6_max)
// MAGIC     ax.set_xlabel('X')
// MAGIC     ax.set_ylabel('Y')
// MAGIC     ax.set_zlabel('Z')
// MAGIC
// MAGIC     plt.show()

// COMMAND ----------

def saveSample(density : DensityHistogram, sampleSize : Int, limitsPath : String, samplePath : String, seed : Long) = {

  val limits : Array[Double] = Array(
    density.tree.rootCell.low(0),
    density.tree.rootCell.high(0),
    density.tree.rootCell.low(1),
    density.tree.rootCell.high(1),
  )
  Array(limits).toIterable.toSeq.toDS.write.mode("overwrite").parquet(limitsPath)

  val rng : UniformRandomProvider = RandomSource.XO_RO_SHI_RO_128_PP.create(seed)
  val sample = density.sample(rng, sampleSize).map(_.toArray)

  var arr : Array[Double] = new Array(2 * sample.length)
  for (i <- 0 until sample.length) {
      arr(2*i + 0) = sample(i)(0)
      arr(2*i + 1) = sample(i)(1)
  }
 
  Array(arr).toIterable.toSeq.toDS.write.mode("overwrite").parquet(samplePath)
}

// COMMAND ----------

def saveSampleCoverageColoring(density : DensityHistogram, sampleSize : Int, limitsPath : String, samplePath : String, seed : Long) = {

  val limits : Array[Double] = Array(
    density.tree.rootCell.low(0),
    density.tree.rootCell.high(0),
    density.tree.rootCell.low(1),
    density.tree.rootCell.high(1),
  )
  Array(limits).toIterable.toSeq.toDS.write.mode("overwrite").parquet(limitsPath)

  val coverageRegions : TailProbabilities = density.tailProbabilities
  val rng : UniformRandomProvider = RandomSource.XO_RO_SHI_RO_128_PP.create(seed)
  val sample = density.sample(rng, sampleSize).map(_.toArray)

  var arr : Array[Double] = new Array(3 * sample.length)
  for (i <- 0 until sample.length) {
      arr(3*i + 0) = sample(i)(0)
      arr(3*i + 1) = sample(i)(1)
      arr(3*i + 2) = 1 - coverageRegions.query(Vectors.dense(sample(i)))
  }
 
  Array(arr).toIterable.toSeq.toDS.write.mode("overwrite").parquet(samplePath)
} 

// COMMAND ----------

// MAGIC %python
// MAGIC
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
// MAGIC def plotSample3D(z_max, limitsPath, samplePath):
// MAGIC
// MAGIC     matplotlib_inline.backend_inline.set_matplotlib_formats('png2x')
// MAGIC
// MAGIC     limits = np.array(spark.read.parquet(limitsPath).collect())[-1,-1]
// MAGIC     values = np.array(spark.read.parquet(samplePath).collect())[-1,-1]
// MAGIC
// MAGIC     x4_min = limits[0]
// MAGIC     x4_max = limits[1]
// MAGIC     x6_min = limits[2]
// MAGIC     x6_max = limits[3]
// MAGIC
// MAGIC     fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
// MAGIC
// MAGIC     length = int(len(values) / 2)
// MAGIC     x = np.ndarray(shape=length)
// MAGIC     y = np.ndarray(shape=length)
// MAGIC     
// MAGIC     for i in range(length):
// MAGIC         x[i] = values[2*i + 0]
// MAGIC         y[i] = values[2*i + 1]
// MAGIC
// MAGIC     ax.scatter(x, y, zs=0, zdir='z', c='blue', alpha = 0.01)
// MAGIC
// MAGIC     # Customize the z axis.
// MAGIC     ax.set_zlim(0.0, z_max)
// MAGIC     ax.set_xlim(x4_min, x4_max)
// MAGIC     ax.set_ylim(x6_min, x6_max)
// MAGIC     ax.set_xlabel('X1')
// MAGIC     ax.set_ylabel('X2')
// MAGIC     #ax.set_zlabel('f_n(X1,X2)')
// MAGIC
// MAGIC     plt.show()

// COMMAND ----------

// MAGIC %python
// MAGIC from matplotlib import cbook
// MAGIC from matplotlib import cm
// MAGIC from matplotlib import colors
// MAGIC from matplotlib.colors import LightSource
// MAGIC import matplotlib.pyplot as plt
// MAGIC from matplotlib.ticker import LinearLocator
// MAGIC import matplotlib_inline.backend_inline
// MAGIC from matplotlib.patches import Rectangle, PathPatch
// MAGIC import mpl_toolkits.mplot3d.art3d as art3d
// MAGIC from matplotlib.transforms import Bbox
// MAGIC import numpy as np
// MAGIC
// MAGIC def plotSampleCoverage(z_max, limitsPath, samplePath, alph):
// MAGIC
// MAGIC     matplotlib_inline.backend_inline.set_matplotlib_formats('png2x')
// MAGIC
// MAGIC     limits = np.array(spark.read.parquet(limitsPath).collect())[-1,-1]
// MAGIC     values = np.array(spark.read.parquet(samplePath).collect())[-1,-1]
// MAGIC
// MAGIC     x4_min = limits[0]
// MAGIC     x4_max = limits[1]
// MAGIC     x6_min = limits[2]
// MAGIC     x6_max = limits[3]
// MAGIC
// MAGIC     fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
// MAGIC
// MAGIC     length = int(len(values) / 3)
// MAGIC     x = np.ndarray(shape=length)
// MAGIC     y = np.ndarray(shape=length)
// MAGIC     col = np.ndarray(shape=length)
// MAGIC     
// MAGIC     for i in range(length):
// MAGIC         x[i] = values[3*i + 0]
// MAGIC         y[i] = values[3*i + 1]
// MAGIC         col[i] = values[3*i + 2]
// MAGIC
// MAGIC     ax.scatter(x, y, zs=0, zdir='z', c=col, cmap=cm.hot, alpha = alph)
// MAGIC
// MAGIC     fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=0, vmax=1), cmap=cm.hot), ax=ax)
// MAGIC
// MAGIC     # Customize the z axis.
// MAGIC     ax.set_zlim(0.0, z_max)
// MAGIC     ax.set_xlim(x4_min, x4_max)
// MAGIC     ax.set_ylim(x6_min, x6_max)
// MAGIC     ax.set_xlabel('X1')
// MAGIC     ax.set_ylabel('X2')
// MAGIC
// MAGIC     plt.show()

// COMMAND ----------

def conditionaDensityPoints(density : DensityHistogram, sliceAxis : Depth, sliceP : Double) : Array[(Double, Double)] = {
  var sliceAxes : Vector[Int] = Vector(sliceAxis)
  var slicePoint : Vector[Double] = Vector(sliceP)
  val splitOrder = tree.splitOrderToDepth(100)
  var conditional : DensityHistogram = quickSlice(density, sliceAxes, slicePoint, splitOrder).normalize
  val leaves : Vector[NodeLabel] = conditional.densityMap.truncation.leaves
  var graphPoints : Array[(Double, Double)] = new Array(2 * leaves.length)
  for (i <- 0 until leaves.length) {
    val leaf : NodeLabel = leaves(i)
    val rect : Rectangle = conditional.tree.cellAt(leaf)
    val densityValue : Double = conditional.densityMap.vals(i)._1

    /* Add slight shift so the display function works */
    graphPoints(2*i) = (rect.low(0) + 0.00001, densityValue)
    graphPoints(2*i + 1) = (rect.high(0) - 0.00001, densityValue)
  }

  graphPoints
}

// COMMAND ----------

def marginalDensityPoints(density : DensityHistogram, keepAxis : Vector[Depth]) : Array[(Double, Double)] = {
 val margin = marginalize(density, keepAxis).normalize
  val leaves : Vector[NodeLabel] = margin.densityMap.truncation.leaves
  var graphPoints : Array[(Double, Double)] = new Array(2 * leaves.length)
  for (i <- 0 until leaves.length) {
    val leaf : NodeLabel = leaves(i)
    val rect : Rectangle = margin.tree.cellAt(leaf)
    val densityValue : Double = margin.densityMap.vals(i)._1

    /* Add slight shift so the display function works */
    graphPoints(2*i) = (rect.low(0) + 0.00001, densityValue)
    graphPoints(2*i + 1) = (rect.high(0) - 0.00001, densityValue)
  }

  graphPoints
}

// COMMAND ----------

def coverageRegionsPrint(distribution : MixtureDistribution, density : DensityHistogram, probabilities : Array[Double], sampleSize : Int, seed : Long) = {
  val dimensions = distribution.lineMixtures(0).start.length
  val sample = normalVectorRDD(spark.sparkContext, sampleSize, dimensions, 4, seed).mapPartitions(iter => distribution.sample(iter)).collect
  
  val p = probabilities.sorted
  val coverageRegions : TailProbabilities = density.tailProbabilities

  val allBins : Array[(Double, Count)] = coverageRegions.tails.vals.sorted.map((_, 0L)).toArray
  val bins : Array[(Double, Count)] = new Array(probabilities.length)

  var bin = 0
  var i = 0
  while (i < allBins.length && bin < bins.length) {
    if (p(bin) <= allBins(i)._1) {
      bins(bin) = (allBins(i)._1, 0L)
      bin += 1
    }
    i += 1
  }
  
  for (i <- 0 until sample.length) {
    val coverageRegionProb : Double = coverageRegions.query(sample(i))
    for (k <- 0 until bins.length) {
      if (coverageRegionProb <= bins(k)._1) {
        bins(k) = (bins(k)._1, bins(k)._2 + 1)
      }
    }
  }

  val proportions : Array[(Double, Double)] = bins.map(t => (t._1, t._2.toDouble / sampleSize))
  for (i <- 0 until bins.length) {
    println("(wantedProbability, achievedProbability, Proportion): (" + p(i) + ", " + proportions(i)._1 + ", " + proportions(i)._2 + ")" )
  }
}

// COMMAND ----------

val limitsPath = s"${rootPath}/limits"
val plotValuesPath = s"${rootPath}/plotValues"
val pointsPerAxis = 256
savePlotValues(density, density.tree.rootCell, pointsPerAxis, limitsPath, plotValuesPath)

// COMMAND ----------

// MAGIC %python
// MAGIC plotDensity(256, 0.05, path + "limits", path + "plotValues")

// COMMAND ----------

// MAGIC %python
// MAGIC plotDensityAndPlane(256, 0.05, path + "limits", path + "plotValues", 0, 3.0)

// COMMAND ----------

// MAGIC %python
// MAGIC plotConditionalDensity3D(256, 0.05, path + "limits", path + "plotValues", 0, 3.0)

// COMMAND ----------

val graphPoints = conditionaDensityPoints(density, 0, 3.0)
implicit val ordering : Ordering[NodeLabel] = leftRightOrd
display(graphPoints.toVector.toDS)

// COMMAND ----------

// MAGIC %python
// MAGIC plotDensityAndPlane(256, 0.05, path + "limits", path + "plotValues", 1, 3.0)

// COMMAND ----------

// MAGIC %python
// MAGIC plotConditionalDensity3D(256, 0.05, path + "limits", path + "plotValues", 1, 3.0)

// COMMAND ----------

val graphPoints = conditionaDensityPoints(density, 1, 3.0)
implicit val ordering : Ordering[NodeLabel] = leftRightOrd
display(graphPoints.toVector.toDS)

// COMMAND ----------

val sampleSize : Int = 10000000
val seed : Long = 91185312L
coverageRegionsPrint(mixture, density, Array(0.9, 0.95, 0.99, 0.999), sampleSize, seed)

// COMMAND ----------

val limitsPath = s"${rootPath}/limits_cov"
val plotValuesPath = s"${rootPath}/plotValues_cov"
val pointsPerAxis = 256
savePlotValuesCoverage(density, density.tree.rootCell, 0.50, pointsPerAxis, limitsPath, plotValuesPath)

// COMMAND ----------

// MAGIC %python
// MAGIC plotDensity(256, 0.05, path + "limits_cov", path + "plotValues_cov")

// COMMAND ----------

savePlotValuesCoverage(density, density.tree.rootCell, 0.90, pointsPerAxis, limitsPath, plotValuesPath)

// COMMAND ----------

// MAGIC %python
// MAGIC plotDensity(256, 0.05, path + "limits_cov", path + "plotValues_cov")

// COMMAND ----------

savePlotValuesCoverage(density, density.tree.rootCell, 0.95, pointsPerAxis, limitsPath, plotValuesPath)

// COMMAND ----------

// MAGIC %python
// MAGIC plotDensity(256, 0.05, path + "limits_cov", path + "plotValues_cov")

// COMMAND ----------

val limitsPath = s"${rootPath}/limits_cov"
val supportPath = s"${rootPath}/support_cov"
saveSupportPlot(density, density.tree.rootCell, 0.5, pointsPerAxis, limitsPath, supportPath)

// COMMAND ----------

// MAGIC %python
// MAGIC plotSupport(0.05, path + "limits_cov", path + "support_cov")

// COMMAND ----------

val limitsPath = s"${rootPath}/limits_cov"
val plotValuesPath = s"${rootPath}/support_cov"
saveSupportPlot(density, density.tree.rootCell, 0.9, pointsPerAxis, limitsPath, supportPath)

// COMMAND ----------

// MAGIC %python
// MAGIC plotSupport(0.05, path + "limits_cov", path + "support_cov")

// COMMAND ----------

val limitsPath = s"${rootPath}/limits_cov"
val plotValuesPath = s"${rootPath}/support_cov"
saveSupportPlot(density, density.tree.rootCell, 0.95, pointsPerAxis, limitsPath, supportPath)

// COMMAND ----------

// MAGIC %python
// MAGIC plotSupport(0.05, path + "limits_cov", path + "support_cov")

// COMMAND ----------

val limitsPath = s"${rootPath}/limits_cov"
val plotValuesPath = s"${rootPath}/plotValues_cov"
savePlotValuesCoverage(density, density.tree.rootCell, 0.99, pointsPerAxis, limitsPath, plotValuesPath)

// COMMAND ----------

// MAGIC %python
// MAGIC plotDensity(256, 0.05, path + "limits_cov", path + "plotValues_cov")

// COMMAND ----------

val limitsPath = s"${rootPath}/limits_cov"
val plotValuesPath = s"${rootPath}/plotValues_cov"
savePlotValuesCoverage(density, density.tree.rootCell, 0.999, pointsPerAxis, limitsPath, plotValuesPath)

// COMMAND ----------

// MAGIC %python
// MAGIC plotDensity(256, 0.05, path + "limits_cov", path + "plotValues_cov")

// COMMAND ----------

val limitsPath = s"${rootPath}/limits_cov"
val supportPath = s"${rootPath}/support_cov"
saveSupportPlot(density, density.tree.rootCell, 0.5, limitsPath, supportPath)

// COMMAND ----------

// MAGIC %python
// MAGIC supportPlot(0.05, path + "limits_cov", path + "support_cov")

// COMMAND ----------

val limitsPath = s"${rootPath}/limits_cov"
val supportPath = s"${rootPath}/support_cov"
saveSupportPlot(density, density.tree.rootCell, 0.9, limitsPath, supportPath)

// COMMAND ----------

// MAGIC %python
// MAGIC supportPlot(0.05, path + "limits_cov", path + "support_cov")

// COMMAND ----------

val limitsPath = s"${rootPath}/limits_cov"
val supportPath = s"${rootPath}/support_cov"
saveSupportPlot(density, density.tree.rootCell, 0.95, limitsPath, supportPath)

// COMMAND ----------

// MAGIC %python
// MAGIC supportPlot(0.05, path + "limits_cov", path + "support_cov")

// COMMAND ----------

val limitsPath = s"${rootPath}/limits_cov"
val supportPath = s"${rootPath}/support_cov"
saveSupportPlot(density, density.tree.rootCell, 0.99, limitsPath, supportPath)

// COMMAND ----------

// MAGIC %python
// MAGIC supportPlot(0.05, path + "limits_cov", path + "support_cov")

// COMMAND ----------

val limitsPath = s"${rootPath}/limits_cov"
val supportPath = s"${rootPath}/support_cov"
saveSupportPlot(density, density.tree.rootCell, 0.999, limitsPath, supportPath)

// COMMAND ----------

// MAGIC %python
// MAGIC supportPlot(0.05, path + "limits_cov", path + "support_cov")

// COMMAND ----------

val graphPoints = marginalDensityPoints(density, Vector(0))
implicit val ordering : Ordering[NodeLabel] = leftRightOrd
display(graphPoints.toVector.toDS)

// COMMAND ----------

val graphPoints = marginalDensityPoints(density, Vector(1))
implicit val ordering : Ordering[NodeLabel] = leftRightOrd
display(graphPoints.toVector.toDS)

// COMMAND ----------

val seed : Long = 123463
val limitsPath = s"${rootPath}/limits"
val samplePath = s"${rootPath}/sample"
saveSample(density, 10000, limitsPath, samplePath, seed)

// COMMAND ----------

// MAGIC %python
// MAGIC plotSample3D(0.5, path + "limits", path + "sample")

// COMMAND ----------

val seed : Long = 5553463
val limitsPath = s"${rootPath}/limits"
val samplePath = s"${rootPath}/sample"
saveSampleCoverageColoring(density, 100, limitsPath, samplePath, seed)

// COMMAND ----------

// MAGIC %python
// MAGIC path = "dbfs:/Users/sandstedt225@gmail.com/data/100GB/crossed_mixture/2/"
// MAGIC plotSampleCoverage(0.5, path + "limits", path + "sample", 1)

// COMMAND ----------

val seed : Long = 5553463
val limitsPath = s"${rootPath}/limits"
val samplePath = s"${rootPath}/sample"
saveSampleCoverageColoring(density, 1000, limitsPath, samplePath, seed)

// COMMAND ----------

// MAGIC %python
// MAGIC path = "dbfs:/Users/sandstedt225@gmail.com/data/100GB/crossed_mixture/2/"
// MAGIC plotSampleCoverage(0.5, path + "limits", path + "sample", 0.5)

// COMMAND ----------

val seed : Long = 5553463
val limitsPath = s"${rootPath}/limits"
val samplePath = s"${rootPath}/sample"
saveSampleCoverageColoring(density, 10000, limitsPath, samplePath, seed)

// COMMAND ----------

// MAGIC %python
// MAGIC path = "dbfs:/Users/sandstedt225@gmail.com/data/100GB/crossed_mixture/2/"
// MAGIC plotSampleCoverage(0.5, path + "limits", path + "sample", 0.2)

// COMMAND ----------

val seed : Long = 5553463
val limitsPath = s"${rootPath}/limits"
val samplePath = s"${rootPath}/sample"
saveSampleCoverageColoring(density, 100000, limitsPath, samplePath, seed)

// COMMAND ----------

// MAGIC %python
// MAGIC path = "dbfs:/Users/sandstedt225@gmail.com/data/100GB/crossed_mixture/2/"
// MAGIC plotSampleCoverage(0.5, path + "limits", path + "sample", 0.1)

// COMMAND ----------

val seed : Long = 5553463
val limitsPath = s"${rootPath}/limits"
val samplePath = s"${rootPath}/sample"
saveSampleCoverageColoring(density, 1000000, limitsPath, samplePath, seed)

// COMMAND ----------

// MAGIC %python
// MAGIC path = "dbfs:/Users/sandstedt225@gmail.com/data/100GB/crossed_mixture/2/"
// MAGIC plotSampleCoverage(0.5, path + "limits", path + "sample", 0.01)
