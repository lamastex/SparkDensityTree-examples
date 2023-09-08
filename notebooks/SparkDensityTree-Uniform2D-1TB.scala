// Databricks notebook source
// MAGIC %md
// MAGIC
// MAGIC ## 1TB 2-dimensional Uniform density estimation
// MAGIC
// MAGIC This project was partly supported by Combient Mix AB through 2023 summer internship in Data Engineering Sciences to Axel Sandstedt and a grant from Wallenberg AI, Autonomous Systems and Software Program funded by Knut and Alice Wallenberg Foundation to Raazesh Sainudiin.
// MAGIC
// MAGIC This notebook was used to time a 1TB dataset estimation coming from a 2 dimensional uniform distribution. The cluster configuration used can be seen below.
// MAGIC
// MAGIC #### Packages used:
// MAGIC #####Needed for RNG/sampling inside library
// MAGIC - commons_rng_client_api_1_5.jar
// MAGIC - commons_rng_core_1_5.jar
// MAGIC - commons_rng_sampling_1_5.jar
// MAGIC - commons_rng_simple_1_5.jar
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

import math.{min, max, abs}
import scala.collection.immutable.{Vector => Vec}
import scala.math.BigInt
import java.math.BigInteger

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.util.LongAccumulator
import org.apache.spark.sql.{ SparkSession }
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Dataset
import org.apache.spark.storage.StorageLevel

import org.apache.spark.mllib.random.RandomRDDs.uniformVectorRDD
import org.apache.spark.mllib.linalg.{ Vector => MLVector, Vectors }

import org.apache.commons.rng.UniformRandomProvider
import org.apache.commons.rng.simple.RandomSource

import co.wiklund.disthist._
import co.wiklund.disthist.Types._
import co.wiklund.disthist.SpatialTreeFunctions._
import co.wiklund.disthist.MergeEstimatorFunctions._
import co.wiklund.disthist.HistogramFunctions._
import co.wiklund.disthist.MDEFunctions._
import co.wiklund.disthist.LeafMapFunctions._

// COMMAND ----------

// MAGIC %md
// MAGIC We set the number of partitions used to something appropriate.
// MAGIC Why do we need **numCores**? This will be used in the last stage in the MDE search, so it needs to be set.

// COMMAND ----------

val numCores = 256
val partsScale = 256
val numTrainingPartitions = 64*partsScale
val numValidationPartitions = 16*partsScale

// COMMAND ----------

// MAGIC %md
// MAGIC #### IMPORTANT
// MAGIC Set the **pathToDataFolder** to your folder of choice.

// COMMAND ----------

val rootPath = "dbfs:/Users/sandstedt225@gmail.com/data/1TB/UNIFORM_2D/"

val dimensions = 2
//val sizeExp = 9
//val trainSize : Long = math.pow(10, sizeExp).toLong
//val validationSize : Long = trainSize / 2
val trainSize : Long = (math.pow(10.toLong,11) * (7.0/16.0)).toLong
val validationSize : Long = (math.pow(10.toLong,11) * (3.0/16.0)).toLong
val trainingPath = rootPath + "countedTrain"
val validationPath = rootPath + "countedTest"

val treePath = rootPath + "spatialTree"
val finestResDepthPath = rootPath + "finestRes"
 
val histogramPath = rootPath + "trainedHistogram"
val finestHistPath = rootPath + "finestHist"
val mdeHistPath = rootPath + "mdeHist"

// COMMAND ----------

// MAGIC %md
// MAGIC Next, we setup the mixture sample RDDs

// COMMAND ----------

val trainingSeed = 12305689
val validationSeed = 546569434
val trainingRDD = uniformVectorRDD(spark.sparkContext, trainSize, dimensions, numTrainingPartitions, trainingSeed)
val validationRDD =  uniformVectorRDD(spark.sparkContext, validationSize, dimensions, numValidationPartitions, validationSeed)

// COMMAND ----------

// MAGIC %md
// MAGIC First up is deriving the box hull of the validation and training data which will be our **root regular raving**.

// COMMAND ----------

/* Get boxhull of training data and test data */
var rectTrain = RectangleFunctions.boundingBox(trainingRDD)
var rectValidation = RectangleFunctions.boundingBox(validationRDD)
val rootBox = RectangleFunctions.hull(rectTrain, rectValidation)

val tree = widestSideTreeRootedAt(rootBox)
Vector(tree.rootCell.low, tree.rootCell.high).toIterable.toSeq.toDS.write.mode("overwrite").parquet(treePath) 
    
/* Hueristic didn't work, floating point precision made the volume so small that it become negative, we set a fixed depth instead */
val finestResDepth = 60
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



// COMMAND ----------

val minimumCountLimit = 100000
val countedTrain = spark.read.parquet(trainingPath).as[(NodeLabel, Count)].rdd
val maxLeafCount = countedTrain.map(_._2).reduce(max(_,_))
println("Max is count is " + maxLeafCount + " at depth " + finestResDepth)
val countLimit = max(minimumCountLimit, maxLeafCount)

//val countLimit = getCountLimit(countedTrain, minimumCountLimit)

// COMMAND ----------



// COMMAND ----------

implicit val ordering : Ordering[NodeLabel] = leftRightOrd
val sampleSizeHint = 1000
val partitioner = new SubtreePartitioner(numTrainingPartitions, countedTrain, sampleSizeHint)
val depthLimit = partitioner.maxSubtreeDepth
val subtreeRDD = countedTrain.repartitionAndSortWithinPartitions(partitioner)
val finestHistogram : Histogram = mergeLeavesHistogram(tree, subtreeRDD, countLimit, depthLimit)

// COMMAND ----------



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

mdeHist.counts.toIterable.toSeq.map(t => (t._1.lab.bigInteger.toByteArray, t._2)).toDS.write.mode("overwrite").parquet(mdeHistPath)
val density = toDensityHistogram(mdeHist).normalize

// COMMAND ----------

// MAGIC %md
// MAGIC The \\(L_1\\)-error is derived below

// COMMAND ----------

val uniformDensity = 1.0
val l1Error = density.densityMap.vals.map{ case (dens, vol) => math.abs(vol*(dens - uniformDensity))}.sum + (1.0 - tree.volumeTotal)
println(l1Error)

// COMMAND ----------


