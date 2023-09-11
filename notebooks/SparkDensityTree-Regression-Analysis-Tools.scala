// Databricks notebook source
// MAGIC %md
// MAGIC ## **Regression Analysis**: Some common tools
// MAGIC
// MAGIC This project was partly supported by Combient Mix AB through 2023 summer internship in Data Engineering Sciences to Axel Sandstedt and a grant from Wallenberg AI, Autonomous Systems and Software Program funded by Knut and Alice Wallenberg Foundation to Raazesh Sainudiin.
// MAGIC
// MAGIC We go through how to apply some common regression analysis tools within the **SparkDensityTree** library. The tools under consideration are: conditional densities, marginalization of densities, highest density regions, sampling and prediction.

// COMMAND ----------

// MAGIC %md
// MAGIC #### Reconstructing the estimate
// MAGIC Lets start by importing in the saved histogram. Please copy over the `pathToDataFolder` value of your choice here!

// COMMAND ----------

/* TODO: write a folder to save data to */
val pathToDataFolder = "dbfs:/users/sandstedt/notebooks"

// COMMAND ----------

import co.wiklund.disthist._
import co.wiklund.disthist.Types._
import co.wiklund.disthist.SpatialTreeFunctions._
import co.wiklund.disthist.MergeEstimatorFunctions._
import co.wiklund.disthist.HistogramFunctions._
import co.wiklund.disthist.MDEFunctions._
import co.wiklund.disthist.LeafMapFunctions._

val rootPath = s"${pathToDataFolder}/introduction"
val treePath = rootPath + "spatialTree"
val finestResDepthPath = rootPath + "finestRes" 
val mdeHistPath = rootPath + "mdeHist"

// COMMAND ----------

// MAGIC %md
// MAGIC Next we load in the `WidestSplitTree` used in our histogram. We also load the `finestResDepth` which we will utilize again later.

// COMMAND ----------

val treeVec : Array[Vector[Double]] = spark.read.parquet(treePath).as[Vector[Double]].collect
  val lowArr : Array[Double] = new Array(5)
  val highArr : Array[Double] = new Array(5)
  for (j <- 0 to 4) {
    if(treeVec(0)(j) < treeVec(1)(j)) {
      lowArr(j) = treeVec(0)(j)
      highArr(j) = treeVec(1)(j)
    } else {
      lowArr(j) = treeVec(1)(j)
      highArr(j) = treeVec(0)(j)
    }
  }

  val tree = widestSideTreeRootedAt(Rectangle(lowArr.toVector, highArr.toVector))

val finestResDepth : Int = spark.read.parquet(finestResDepthPath).as[Depth].collect()(0) 

// COMMAND ----------

// MAGIC %md
// MAGIC We load in the leaves, or cells, in our histogram and reconstruct the `Histogram`.

// COMMAND ----------

val mdeCounts : Array[(NodeLabel, Count)] = spark.read.parquet(mdeHistPath).as[(NodeLabel, Count)].collect

// COMMAND ----------

val mdeHist : Histogram = Histogram(tree, mdeCounts.map(_._2).reduce(_+_), fromNodeLabelMap(mdeCounts.toMap))

// COMMAND ----------

// MAGIC %md
// MAGIC Now we reconstruct the density estimate.

// COMMAND ----------

val density = toDensityHistogram(mdeHist).normalize

// COMMAND ----------

// MAGIC %md
// MAGIC #### Getting conditionals using quickSlice
// MAGIC Since the true underlying density \\( f \\) comes from a multivariate standard normal \\(\mathbf{X} = (X_1,\dots,X_5)\\), we have that conditioning on all dimensions at 0.0 except one gives a new \\( N(0,1) \\) density 
// MAGIC $$f(x \big| X_2=0, \dots, X_5=0).$$
// MAGIC We inspect the conditional of our estimate and the same point. `sliceAxes` contains the indices for the dimensions we wish to condition on. `slicePoint` contains points along those dimensions 
// MAGIC which define where or on what values we condition on. Note that an index of 0 corresponds to \\(X_1\\), an index of 1 corresponds to \\(X_2\\), and so on.

// COMMAND ----------

var sliceAxes : Vector[Int] = Vector(1, 2, 3, 4)
var slicePoint : Vector[Double] = Vector(0.0, 0.0, 0.0, 0.0)

// COMMAND ----------

// MAGIC %md
// MAGIC We want to use the optimised conditional function in the library, so we will need to fill in some data which will be passed to the method.

// COMMAND ----------

// MAGIC %md
// MAGIC The first thing we need to generate is a split order for our root box. Making sure that we generate enough splits, you should pass in the `finestResDepth` value from when you constructed the density, or just pass in some very large value deeper than all your leaves.

// COMMAND ----------

val splitOrder = tree.splitOrderToDepth(finestResDepth)

// COMMAND ----------

// MAGIC %md
// MAGIC Next up, if you know that you will be retrieving several conditionals, you can **optionally** setup two **buffers** for reuse in the function, removing two large heap 
// MAGIC allocations within the function. We set them up for completeness sake. These buffers will be used to hold the new conditional leaves and values inside the function, so
// MAGIC we only have to make them as large as our density's leaf vector to know that we will not run out of memory inside them.

// COMMAND ----------

var sliceLeavesBuf : Array[NodeLabel] = new Array(density.densityMap.truncation.leaves.length)
var sliceValuesBuf : Array[(Double,Volume)] = new Array(density.densityMap.truncation.leaves.length)

// COMMAND ----------

// MAGIC %md
// MAGIC Now we can apply quickSlice and hopefully get the conditional density

// COMMAND ----------

var conditional : DensityHistogram = quickSlice(density, sliceAxes, slicePoint, splitOrder, sliceLeavesBuf, sliceValuesBuf)

// COMMAND ----------

// MAGIC %md
// MAGIC We need to check the validity of the conditional: If we did not slice any leaves, and thus constructed a conditional on a 0-probability region, it is ill-defined. If everything went well, we **normalize** it.

// COMMAND ----------

if (conditional != null) {
  conditional = conditional.normalize
}

// COMMAND ----------

// MAGIC %md
// MAGIC We display the conditional density.

// COMMAND ----------

val leaves : Vector[NodeLabel] = conditional.densityMap.truncation.leaves
var graphPoints : Array[(Double, Double)] = new Array(2 * leaves.length)
for (i <- 0 until leaves.length) {
  val leaf : NodeLabel = leaves(i)
  val rect : Rectangle = conditional.tree.cellAt(leaf)
  val densityValue : Double = conditional.densityMap.vals(i)._1

  /* Add slight shift so the display function works */
  graphPoints(2*i) = (rect.low(0) + 0.001, densityValue)
  graphPoints(2*i + 1) = (rect.high(0) - 0.001, densityValue)
}

// COMMAND ----------

display(graphPoints.toVector.toDS)

// COMMAND ----------

// MAGIC %md
// MAGIC Getting conditionals on any other dimensions works exactly the same as above. We may consider the conditional estimate for 
// MAGIC $$f(x_2, x_5 \big| X_1=0.1, X_3=0.3, X_4 = 0.4).$$
// MAGIC We reuse the buffers and split order we previously generated and construct a new conditional density:

// COMMAND ----------

sliceAxes = Vector(0, 2, 3)
slicePoint = Vector(0.1, 0.3, 0.4)
val conditional25 = quickSlice(density, sliceAxes, slicePoint, splitOrder, sliceLeavesBuf, sliceValuesBuf)

// COMMAND ----------

// MAGIC %md
// MAGIC #### Getting the marginal distribution for a set of axes
// MAGIC Marginalising a density is similar to slicing, but instead of providing a vector of axes you which to remove (condition on), you provide the axes that you wish to keep. We generate and display a 1D marginal density
// MAGIC $$f_4(x) = \int_\mathbb{R} \int_\mathbb{R} \int_\mathbb{R} \int_\mathbb{R} f(x_1,x_2,x_3,x,x_5)dx_1 dx_2 dx_3 dx_5$$

// COMMAND ----------

val axesToKeep4 : Vector[Axis] = Vector(3)
val margin4 = marginalize(density, axesToKeep4).normalize

// COMMAND ----------

val leaves4 : Vector[NodeLabel] = margin4.densityMap.truncation.leaves
var graphPoints4 : Array[(Double, Double)] = new Array(2 * leaves4.length)
for (i <- 0 until leaves4.length) {
  val leaf : NodeLabel = leaves4(i)
  val rect : Rectangle = margin4.tree.cellAt(leaf)
  val densityValue : Double = margin4.densityMap.vals(i)._1

  /* Add slight shift so the display function works */
  graphPoints4(2*i) = (rect.low(0) + 0.001, densityValue)
  graphPoints4(2*i + 1) = (rect.high(0) - 0.001, densityValue)
}

// COMMAND ----------

display(graphPoints4.toVector.toDS)

// COMMAND ----------

// MAGIC %md
// MAGIC Similarly for multivariate margins, say we want to keep axes 1,3 and 5:
// MAGIC $$f_{1,3,5}(x,y,z) = \int_\mathbb{R} \int_\mathbb{R} f(x,x_2,y,x_4,z)dx_2 dx_4$$

// COMMAND ----------

val axesToKeep135 : Vector[Axis] = Vector(0,2,4)
val margin135 = marginalize(density, axesToKeep135).normalize

// COMMAND ----------

// MAGIC %md
// MAGIC #### Getting coverage regions using the estimate
// MAGIC One thing of interest in regression analysis is to calculate **highest density regions**. Our estimate provides a simple way of calculating highest density regions for many
// MAGIC probabilities.
// MAGIC Suppose that all leaves are sorted \\(l_1,\dots,l_n\\) according to their density values high to low. Let \\(p_k\\) denote the probability of leaf \\(l_k\\). The highest density regions we retrieve will then look like: 
// MAGIC
// MAGIC $$\bigg[\ [ \\{l_1\\}, p_1 ], [ \\{l_1, l_2\\}, p_1 + p_2 ], \dots,[\\{l_k : 1 \leq k \leq n\\}, \sum_{k=1}^n p_k]\ \bigg]$$
// MAGIC
// MAGIC Suppose that we are interesed in the smallest region with probability \\(\geq\\) 0.95. 

// COMMAND ----------

val wantedProbability : Double = 0.95

// COMMAND ----------

// MAGIC %md
// MAGIC The generated highest density regions can be found in `highestDensityRegions`.

// COMMAND ----------

  val highestDensityRegions : TailProbabilities = conditional.tailProbabilities

// COMMAND ----------

// MAGIC %md
// MAGIC To get the smallest region with a given minimum probability, we can call the method `confidenceRegion`. The method returns the
// MAGIC region's probability.

// COMMAND ----------

val actualProbability : Double = highestDensityRegions.confidenceRegion(wantedProbability)
assert(actualProbability >= wantedProbability)
println(actualProbability)

// COMMAND ----------

// MAGIC %md
// MAGIC Suppose we are wondering if the point 1.0 is contained within our region.
// MAGIC We first convert the point to a Vector and the pass it to the higestDensityRegions query method. The return value `regionProbability` is equal to the probability of the smallest
// MAGIC highest density region in the above collection than contains our point 1.0.

// COMMAND ----------

import org.apache.spark.mllib.linalg.{ Vector => MLVector, _ }
val regionProbability : Double = highestDensityRegions.query(Vectors.dense(1.0))

// COMMAND ----------

// MAGIC %md
// MAGIC To check if the points lies within the 95% region, we simply check if its probability is \\(\leq\\) the region's probability.

// COMMAND ----------

if (regionProbability <= actualProbability) {
  println("The point is within the estimated 95%-highest density region!")
} else {
  println("The point is not within the estimated 95%-highest density region!")
}

// COMMAND ----------

// MAGIC %md
// MAGIC We generate some new
// MAGIC data from the underlying distribution and check how the points are distributed among our highest density regions.

// COMMAND ----------

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.random.RandomRDDs.normalVectorRDD

val numPoints : Int = 100000
val dim : Int = 1
val numPartitions : Int = 2
val seed : Long = 93354446

val testData : Array[MLVector] = normalVectorRDD(spark.sparkContext, numPoints, 1, numPartitions, seed).collect

// COMMAND ----------

/* (Probability, Count) */
var bins : Array[(Double, Count)] = highestDensityRegions.tails.vals.sorted.map((_, 0L)).toArray
for (i <- 0 until testData.length) {
  val regionProb : Double = highestDensityRegions.query(testData(i))
  for (k <- 0 until bins.length) {
    if (regionProb <= bins(k)._1) {
      bins(k) = (bins(k)._1, bins(k)._2 + 1)
    }
  }
}

// COMMAND ----------

// MAGIC %md
// MAGIC We print out each region's probability and the proportion of generated points found within the region:

// COMMAND ----------

val proportions : Array[(Double, Double)] = bins.map(t => (t._1, t._2.toDouble / numPoints))
proportions.foreach(t => println("(Probability, Proportion): (" + t._1 + ", " + t._2 + ")" ))

// COMMAND ----------

// MAGIC %md
// MAGIC **Note** that getting highest density regions for multivariate estimates works exactly the same:

// COMMAND ----------

  val highestDensityRegions25 : TailProbabilities = conditional25.tailProbabilities

// COMMAND ----------

// MAGIC %md
// MAGIC #### Sampling from the estimate and prediction (Requires Apache Commons RNG jars)
// MAGIC In order to sample data from the estimate's distribution, one needs to generate a `UniformRandomProvider`from the Apache Commons RNG library. For guidelines regarding good RandomSources,
// MAGIC see the aforementioned library's documentation.

// COMMAND ----------

import org.apache.commons.rng.UniformRandomProvider
import org.apache.commons.rng.simple.RandomSource

val rng : UniformRandomProvider = RandomSource.XO_RO_SHI_RO_128_PP.create()

// COMMAND ----------

// MAGIC %md
// MAGIC Now, sampling can be done as easily as a one line call. A small warning: if one needs to **generate a sample of large size** from some density, it is much more **efficient** to batch the sample into a single call as is being done below, instead of
// MAGIC calling the sample methods one time for each datapoint. This is because the method requires a costly setup.

// COMMAND ----------

val sampleSize : Int = 1000000
val sample = conditional.sample(rng, sampleSize)

// COMMAND ----------

// MAGIC %md
// MAGIC We display the histogram of the sample from the original conditional density estimate
// MAGIC  just to verify that it follows the estimate's distribution.

// COMMAND ----------

var counts : Array[Double] = new Array(leaves.length)
counts = counts.map(d => 0.0)
val boxes : Vector[Rectangle] = leaves.map(conditional.tree.cellAt(_))

for (i <- 0 until sampleSize) {
  val point : MLVector = Vectors.dense(sample(i).toArray)
  for (j <- 0 until leaves.length) {
    if (boxes(j).contains(point)) {
      counts(j) += 1.0
    }
  }
}

assert(counts.reduce(_+_) == sampleSize)

// COMMAND ----------

for (i <- 0 until leaves.length) {
  val leaf : NodeLabel = leaves(i)
  val rect : Rectangle = conditional.tree.cellAt(leaf)
  /* Add slight shift so the display function works */
  graphPoints(2*i) = (rect.low(0) + 0.001, counts(i) / sampleSize)
  graphPoints(2*i + 1) = (rect.high(0) - 0.001, counts(i) / sampleSize)
}

// COMMAND ----------

display(graphPoints.toVector.toDS)

// COMMAND ----------

// MAGIC %md
// MAGIC Now that we know how to sample from a density, say
// MAGIC $$f(x_2, x_5 \big| X_1, X_3, X_4),$$
// MAGIC if we are given a vector of realized values of some explanatory variables
// MAGIC $$(x_1, x_3, x_4) = (0.1, 0.3, 0.4)$$
// MAGIC then **predicting** a value of (X_2, X_5) given (X_1=x_1, X_3=x_3, X_4=x_4) may be done through a slice and sampling 1 point from the conditional.

// COMMAND ----------

val sliceAxes25 : Vector[Int] = Vector(0,2,3)
val slicePoint25 : Vector[Double] = Vector(0.1, 0.3, 0.4)
val conditional25 = quickSlice(density, sliceAxes25, slicePoint25, splitOrder, sliceLeavesBuf, sliceValuesBuf)
val prediction25 = conditional25.sample(rng, 1)
println(prediction25)

// COMMAND ----------


