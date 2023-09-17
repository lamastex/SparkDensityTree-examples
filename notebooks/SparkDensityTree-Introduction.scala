// Databricks notebook source
// MAGIC %md
// MAGIC ## Distributed Computation of Minimum Distance Histograms
// MAGIC
// MAGIC This project was partly supported by Combient Mix AB through 2023 summer internship in Data Engineering Sciences to Axel Sandstedt and a grant from Wallenberg AI, Autonomous Systems and Software Program funded by Knut and Alice Wallenberg Foundation to Raazesh Sainudiin.
// MAGIC
// MAGIC This Notebook sets out to give a introduction and guide for how to use the [**SparkDensityTree**](https://github.com/lamastex/SparkDensityTree) estimator, a histogram estimator with performance guarantees for any density \\(f\\). For any details regarding the
// MAGIC mathematics, or any of the other theoretical underpinnings, we shall refer to the relevant paper, and as such, will focus on the more practical aspects for constructing an estimate. What challenges that entail we will get into soon.

// COMMAND ----------

// MAGIC %md
// MAGIC #### The Basic Ideas of the Estimator
// MAGIC The basis of the estimator is a Binary Space Partitioning strategy, **Regular Pavings**. A regular paving in \\(\mathbb{R}^d\\) is a recursive data structure consisting of a root box 
// MAGIC $$[\underline{x_{1}},\overline{x_{1}}]\times\cdots\times[\underline{x_{d}},\overline{x_{d}}]$$
// MAGIC and two splitting rules. Any split applied to the data structure must:
// MAGIC 1. be done in the first dimension where the root box have the largest width $$\iota := \min \text{argmax}_{1 \leq k \leq d} \ \overline{x_{k}} - \underline{x_{k}}$$
// MAGIC 2. Any split must happen in the middle of the root box along the axis, and creates two new regular pavings
// MAGIC $$\rho L=[\underline{x_{1}},\overline{x_{1}}]\times\cdots\times\bigg[\underline{x_{\iota}},\frac{\overline{x_{\iota}}-\underline{x_{\iota}}}{2}\bigg)\times\cdots\times[\underline{x_{d}},\overline{x_{d}}]$$
// MAGIC $$\rho R=[\underline{x_{1}},\overline{x_{1}}]\times\cdots\times\bigg[\frac{\overline{x_{\iota}}-\underline{x_{\iota}}}{2},\overline{x_{\iota}}\bigg]\times\cdots\times[\underline{x_{d}},\overline{x_{d}}]$$
// MAGIC If we apply a set of splits on this recursive structure, it is natural to view the structure as a binary tree.
// MAGIC
// MAGIC Defining a histogram through a regular paving is as simple as mapping the leaves of the binary tree to a non-negative count. The general data structure in which we map all the tree's nodes to some set
// MAGIC are called **Mapped Regular Pavings**. **Statistical Regular Pavings** are mapped regular pavings mapping the nodes into \\(\mathbb{Z}_{\geq 0}\\).
// MAGIC
// MAGIC What we seek is a histogram density estimate that minimizes the \\(L_1\\)-distance to the true underlying density \\(f\\). The perhaps most important aspect in this search is to determine the most refined
// MAGIC histogram we are considering. This is where the count limit comes in. The user will set a count limit which the library uses to find the coarsest histogram whose cells contain counts that do not exceed the limit. The count of a cell refers to the number of training datapoints found within the cell. Therefore the count limit is a kind of smoothing parameter.
// MAGIC
// MAGIC Now, the count limit does not fully determine our final estimate; it just defines the most refined histogram we are willing to look at. Special methods is then applied to search in the range of coarser histograms
// MAGIC for a suitable estimate, the minimum distance estimate.
// MAGIC
// MAGIC For understanding the distributed ideas that this library implements, see papers [Scalable Multivariate Histograms](http://lamastex.org/preprints/20180506_SparkDensityTree.pdf), [Scalable Algorithms in Nonparametric Computational Statistics](https://uu.diva-portal.org/smash/record.jsf?aq2=%5B%5B%5D%5D&c=1&af=%5B%5D&searchType=UNDERGRADUATE&sortOrder2=title_sort_asc&language=en&pid=diva2%3A1711540&aq=%5B%5B%7B%22freeText%22%3A%22Scalable+Algorithms+in+Nonparametric%22%7D%5D%5D&sf=all&aqe=%5B%5D&sortOrder=author_sort_asc&onlyFullText=false&noOfRows=50&dswid=-2058) and [Scalable Nonparametric L1 Density Estimation via Sparse Subtree Partitioning](https://github.com/lamastex/2023-mastersthesis-AxelSandstedt).
// MAGIC
// MAGIC For more information about **Regular Pavings**, see [Mapped Regular Pavings](https://interval.louisiana.edu/reliable-computing-journal/volume-16/reliable-computing-16-pp-252-282.pdf).

// COMMAND ----------

// MAGIC %md
// MAGIC #### Strengths and Weaknesses
// MAGIC We first consider some **strengths**:
// MAGIC   * Can estimate any unknown density \\( f \in L_1\\), i.e., the distribution of any continuous random variable \\(X^d\\). 
// MAGIC   * Easily understandable evaluation of performance using the \\(L_1\\)-error \\( \int_{\mathbb{R}} \big|f^* - f\big| \\)
// MAGIC   * Made to be **scalable**, constructed for distributed data sets. 
// MAGIC   * Strict arithmetic imposed by the spltting rules of **_Regular Pavings_** enables us to construct useful tools; we can generate conditional distributions on arbitrary points in any set of arbitrary dimensions.  We can easily construct highest density regions with specific probabilities. Furthermore, we can efficiently sample from the estimate's distribution itself, or any conditional or marginal distribution estimate derived form it.
// MAGIC
// MAGIC Next are some **weaknesses**:
// MAGIC   * Hard to construct an estimate; there are several parameters (with both **statistical** and **practical** implications) which the user has to manually adjust for any specific use case.
// MAGIC   * The construction of the estimate can be seen as a 4-stage process in which 3 of the stages requires the user to get their hands dirty.

// COMMAND ----------

// MAGIC %md
// MAGIC #### Stages Overview
// MAGIC We give a quick overview of the library's 4-stage process to generate a density estimate. It can be described by the following stages:
// MAGIC 1. Root box calculation
// MAGIC 2. Data splitting and labeling
// MAGIC 3. Merging up to count limit
// MAGIC 4. Find minimum distance estimate
// MAGIC
// MAGIC ###### **Stage 1: Calculating the root box**
// MAGIC This stage involves calculating the box hull, or the **root box** of the estimate's **root regular paving**, of our training and validation data. This is of course an extremely parallel task
// MAGIC in which every worker finds the minimum and maximum value in every dimension of its data, and then the minimum and maximum between the workers can be determined. Nonetheless, this is a very time-consuming
// MAGIC stage as we have to work with raw data which entails a floating point value in every dimension.
// MAGIC
// MAGIC ###### **Stage 2: Labeling the data at a very low depth**
// MAGIC The [**SEBTreeMC**](https://link.springer.com/article/10.1007/s42081-019-00054-y) algorithm which gives a randomized splitting procedure using **mapped regular pavings** is what we 
// MAGIC use to generate a path of increasingly more refined histogram densities. Given that we fulfill some assumptions described in the algorithm, the path of histograms almost surely
// MAGIC converge to the underlying density. The idea is that we have all our training data in the root cell. We want all our cells to contain a count no larger than some given count limit.
// MAGIC We continuously split the cell with the largest count, and if several cells are tied for the maximum, we choose one uniformly at random. Any points found in the cell that was just
// MAGIC split are reassigned to one of the two newly created cells depending on which one every point now finds itself in. The algorithm halts as soon as every cell contain no more points
// MAGIC than the allowed limit.
// MAGIC
// MAGIC The algorithm is inherently sequential, and thus we have to work around this splitting procedure somehow. [**Algorithm 7 and 8**](https://uu.diva-portal.org/smash/record.jsf?aq2=%5B%5B%5D%5D&c=1&af=%5B%5D&searchType=UNDERGRADUATE&sortOrder2=title_sort_asc&language=en&pid=diva2%3A1711540&aq=%5B%5B%7B%22freeText%22%3A%22Scalable+Algorithms+in+Nonparametric%22%7D%5D%5D&sf=all&aqe=%5B%5D&sortOrder=author_sort_asc&onlyFullText=false&noOfRows=50&dswid=4324) in the link provides a distributed solution to this problem. 
// MAGIC What we can do instead is to choose a large depth such that every cell at the new depth contain a count less than or equal to the given limit, usually a count of 0 or 1.
// MAGIC A distributed backtracking or merging procedure is then applied in which cells are merged up together until no cells can be merged without going above the count limit.
// MAGIC **Stage 2** refers to the splitting step in which we find for each data point what cell at the given depth the point resides in and find the count for each cell. Since we only care about counts within cells,
// MAGIC when every count has been calculated, the original data is no longer relevant and can be discarded. Furthermore, due to the sparse representation
// MAGIC of **mapped regular pavings** in which we only store leaf nodes, or cells, with nonzero counts, the memory usages becomes very small. The number of leaves we have to represent is bounded by the
// MAGIC number of data points, and as such, we can work with extremely large depths.
// MAGIC
// MAGIC **The user's responsibility** here is to choose an appropriate **depth** to split down to, usually denoted as the **finest resolution depth** `finestResDepth`.
// MAGIC  The name comes from the theoretically perfect choice of depth in which we generate the most coarse histogram in which every cell has a count no larger than 1. 
// MAGIC  Since this is possible to do using our library, we can, at least theoretically, represent as complex densities
// MAGIC as we want. In practice, the user must ensure that the depth they provide is sufficently deep such that no cell exceeds the user's provided **count limit**.
// MAGIC
// MAGIC ###### **Stage 3: Merging to the count limit**
// MAGIC [Chapter 3]() Provides a distributed way of merging the cells up to the limit in a way which minimizes communication between workers. This entails sorting the leaves according to a intuitive left-right
// MAGIC ordering within binary trees, and then find large subtrees of leaves and distribute them evenly between workers using a **SubtreePartitioner**. 
// MAGIC Every worker can then merge its leaves within any of its given subtrees without having to communicate over the network.
// MAGIC
// MAGIC  **The user's responsibilities** here are of both **practical** and **statistical** importance. Starting with the **practical** one, the user needs to provide two parameters with some appropriate 
// MAGIC  values in order for the merging to go smoothly.
// MAGIC
// MAGIC  `sampleSizeHint` determines the amount of points the library will use in order estimate the size of subtrees, or branches, to be sent out to workers during the merging phase. A small value here will result in poor
// MAGIC  load distribution among the workers, so it is a performance parameter. In the worst case, one worker would get to much work and crash due to memory limitations.
// MAGIC
// MAGIC  `subtreePartitions` determines how many partitions to use in the merging task. This parameter is not as simple as setting the number of partitions to the number of cores over the network, but instead
// MAGIC  should be choosen to be large enough for every worker to be assigned small enough partitions that fit in memory. As a small example we have a network of 9 Workers with 2 cores each and 8GB of RAM. 
// MAGIC  When running a 5D Gaussian case with \\(10^8\\) data points and a count limit of 10000 using only 18 partitions, 1 partition for each core, the partitions become to large for any worker to hold in memory.
// MAGIC   Thus an OutOfMemory error will soon be upon us. However, we found that 128 partitions for our 9 workers ran fine, allowing the workings to spill data to disk as needed. 
// MAGIC
// MAGIC  Luckily, from experience I've found that choosing a large number of partitions does not degrade performance due to overhead, and have only been found to have either negligble effect, or even enhance the 
// MAGIC  performance. 
// MAGIC
// MAGIC  `countLimit` Determines the smoothness of the most refined histogram we shall consider when deciding a final estimate among in the path of more and more refined histograms talked about in **Stage 2**.
// MAGIC  As such, it has **statistical performance** implications on the final result. Furthermore, the **practical** implication of the limit is that it determines the range of histograms of which we search.
// MAGIC  This in turn affects the **computational performance** both in time and in memory usage. The time increases as we search between increasingly more complex histograms, and memory usage increases a lot as well
// MAGIC  since the operations which are applied in **Stage 4** will force us to represent some zero-count leaves. This memory issue is currently the limiting factor of the library, and removing it would allow us 
// MAGIC  to reduce the count limit even more.
// MAGIC
// MAGIC ###### **Stage 4: Finding the minimum distance estimate**
// MAGIC In this stage we search within the given range of histograms previously mentioned. Certain operations in this step increases quadratically in cost relative to the number of histograms we consider, and we can
// MAGIC therefore not find the **minimum distance estimate**, or **mde**, within the whole set of histograms. Instead, we begin by choosing \\(k\\) histograms, 
// MAGIC roughly evenly spaced along the path of more and more refined histograms.
// MAGIC We then find the **mde** among the chosen histograms, and "zoom in" on the path around the **mde**. The new range becomes the range of histograms between the two histograms closest to the **mde**
// MAGIC in the set of histograms under consideration during the iteration. In the next iteration, a new set of evenly spaced histograms is chosen, and we continue this effort until we cannot zoom in any more.
// MAGIC The final **mde** becomes the final estimate.
// MAGIC
// MAGIC **The user's responsibility** here is simple; only a choice of \\( k \\) has to be provided. A larger \\( k \\) means that we will consider more histograms and cover a larger part of the path in each
// MAGIC iteration, but the **computational cost** increases quadratically in return. 
// MAGIC
// MAGIC We now move on to actual code.

// COMMAND ----------

// MAGIC %md
// MAGIC We start by importing some parts of the library.

// COMMAND ----------

import co.wiklund.disthist._
import co.wiklund.disthist.Types._
import co.wiklund.disthist.SpatialTreeFunctions._
import co.wiklund.disthist.MergeEstimatorFunctions._
import co.wiklund.disthist.HistogramFunctions._
import co.wiklund.disthist.MDEFunctions._

// COMMAND ----------

// MAGIC %md
// MAGIC We then generate data from a 5D Standard Gaussian \\( N_5(\mathbf{0}, \mathbb{I}) \\).  

// COMMAND ----------

val dimensions : Int = 5
val sizeExp : Int = 7
val numPartitions : Int = 64
val trainSize : Long = math.pow(10, sizeExp).toLong

// COMMAND ----------

// MAGIC %md
// MAGIC `trainingRDD` is the set of points which we will construct histograms from. In **stage 4** we use the set of "held-out" points in our generated training data found in `validationRDD` to choose a minimum distance estimate. In this case, our choice of \\(\varphi\\), the held out proportion, is \\( \varphi = \frac{1}{3} \\). To get an indication of how this affect the end result, we reiterate **Theorem 3** from 
// MAGIC [**_Minimum distance histograms with universal performance guarantees_**](https://link.springer.com/article/10.1007/s42081-019-00054-y):
// MAGIC
// MAGIC **Theorem**: Let \\(0 < \varphi < \frac{1}{2} \\) and \\(n < \infty\\). Let the finite set \\( \Theta \\) determine a class of adaptive multivariate histograms based on statistical regular pavings
// MAGIC with \\(\int f_{n - \varphi n, \theta} = 1\\) for all \\( \theta \in \Theta \\). Let \\(f_{n - \varphi n, \theta^*}\\) be the minimum distance estimate. Then for all \\( n, \varphi n, \theta\\) and 
// MAGIC \\( f \in L_1 \\): 
// MAGIC
// MAGIC $$E\bigg(\int \bigg| f_{n-\varphi n, \theta^*} - f \bigg| \bigg) \leq 3 \min_\theta E\bigg(\int\bigg|f_{n,\theta}-f\bigg|\bigg)\bigg(1+\frac{2\varphi}{1-\varphi}+8\sqrt{\varphi}\bigg)+8\sqrt{\frac{\text{log}2\big|\Theta\big|(\big|\Theta\big|+1)}{\varphi n}}.$$

// COMMAND ----------

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.random.RandomRDDs.normalVectorRDD
import org.apache.spark.mllib.linalg.{ Vector => MLVector, _ }

val trainingRDD : RDD[MLVector] = normalVectorRDD(spark.sparkContext, trainSize, dimensions, numPartitions, 1230568)
val validationSize = trainSize/2
val validationRDD : RDD[MLVector] = normalVectorRDD(spark.sparkContext, validationSize, dimensions, numPartitions, 5465694)

// COMMAND ----------

// MAGIC %md
// MAGIC We have now reached **stage 1** and should determine a **root box** in which our histogram estimates will take values within. 
// MAGIC A natural choice which will contain both the **training data** and the **validation data**
// MAGIC is the box hull of all the data points under consideration. We can retrieve the hull by the following steps:

// COMMAND ----------

var rectTrain : Rectangle = RectangleFunctions.boundingBox(trainingRDD)
var rectValidation : Rectangle = RectangleFunctions.boundingBox(validationRDD)
val rootBox : Rectangle = RectangleFunctions.hull(rectTrain, rectValidation)  

// COMMAND ----------

// MAGIC %md
// MAGIC In **stage 2** we decide a depth at which we count what points goes to which cell. In the end, we will have discarded all our data points and moved over to working with cells consisting of a `NodeLabel` (a string
// MAGIC of bits representing left and right turns in the binary tree of splits to reach the leaf) and a `Count` which refers to the number of points found in that cell. We only store leaves of the binary tree with a non-zero count, which translate to cells in our partitioning of the **root box** which where found to contain a point in them.
// MAGIC
// MAGIC We start by creating a `WidestSplitTree` class object, which simply represents a Regular Paving with the constructed root box and any functionality that comes with it. 

// COMMAND ----------

val tree : WidestSplitTree = widestSideTreeRootedAt(rootBox)

// COMMAND ----------

// MAGIC %md
// MAGIC The specific **depth** one should choose depends fully on the user's data. The user must in some way using a rule-of-thumb determine a suitable depth at which data will be sufficiently sparse among the cells. 
// MAGIC
// MAGIC The rule-of-thumb we use says: Find the smallest depth such that every cell's sides does not exceed a given length, 
// MAGIC `finestResSideLength` in the code. The second line of code calculated the given depth using some intimidating scala code. Now we have our `finestResDepth`.

// COMMAND ----------

val finestResSideLength = 1e-2
val finestResDepth : Int = tree.descendBoxPrime(Vectors.dense(rootBox.low.toArray)).dropWhile(_._2.widths.max > finestResSideLength).head._1.depth

// COMMAND ----------

// MAGIC %md
// MAGIC Now that we have our wanted **depth** we wish to work up from, we must for each data point find what cell it resides in. We then sum up and calculate the final count for our cells, and **stage 2**
// MAGIC is completed. Since points residing in the same cell may be found on different machines at this point in time, we must apply an **all-to-all** communication pattern to do the summation.
// MAGIC
// MAGIC The validation data does not have to be reduced by key; the optimized quickToLabeledNoReduce method should be used.

// COMMAND ----------

var countedTrain : RDD[(NodeLabel, Count)] = quickToLabeled(tree, finestResDepth, trainingRDD).cache
var countedValidation : RDD[(NodeLabel, Count)] = quickToLabeledNoReduce(tree, finestResDepth, validationRDD)

// COMMAND ----------

// MAGIC %md
// MAGIC We have now reached **stage 3**, perhaps the most error-prone part of the library. At this point in the process, we have in storage non-zero **count** cells at some very low **depth**.
// MAGIC We wish to create the coarsest histogram with no cell containing a count _larger than the **count limit**_ defined by the user and/or the data. In our distributed setting this operation can be reduced to a single **all-to-all**
// MAGIC communication pattern, in which we distribute the data between our workers such that every worker **owns** or stores whole **branches** of the tree which it can safely work within.
// MAGIC
// MAGIC In order to **estimate** how the data is distributed between branches of our binary tree, the **driver** need to sample data from all workers. The user's first two responsibilities here is to determine
// MAGIC 1. How many **partitions** should be used in the merging process (_Recommended to be large_),
// MAGIC 2. For each new **partition**, how much data should the **driver** sample?
// MAGIC
// MAGIC Suppose that we decide we want to do the merging operation using 64 **partitions**. What `sampleSizeHint` implies then is that the driver will sample roughly `64*sampleSizeHint` number of
// MAGIC points. In this example, we will make do with a hint of 1000. To small of a hint and our estimation of how the data is distributed among the branches becomes bad and affect **load distribution** among the
// MAGIC workers as a result. In the **worst case**, a worker is given to much work and runs **out of memory**, crashing the program.

// COMMAND ----------

val sampleSizeHint : Int = 1000

// COMMAND ----------

// MAGIC %md
// MAGIC In order to use the library's custom RDD Partitioner, you have to create one using the following parameters:
// MAGIC * `numPartitions` - The number of partitions to be used in the merge operation
// MAGIC * `countedTrain` - Our leaves to be used to construct the histogram
// MAGIC * `sampleSizeHint` - The previous hint
// MAGIC In creating the Partitioner, the driver samples data and estimates the distribution of data among the branches.

// COMMAND ----------

val partitioner : SubtreePartitioner = new SubtreePartitioner(numPartitions, countedTrain, sampleSizeHint)

// COMMAND ----------

// MAGIC %md
// MAGIC Now it is time to apply the **all-to-all** communication pattern. We send data to workers according to how the Partitioner have assigned branches to workers. Each worker then sorts its data locally.
// MAGIC It is at this line of code that **out-of-memory** errors usually occur; If `numPartitions`is to low, the workers are not able to spill data to disk and runs out of memory since the partitions become to large.
// MAGIC Thus a large value of `numPartitions` is recommended. We set the ordering for `NodeLabel` to be the intuitive left-right ordering of leaves within a binary tree so that spark can sort the labels accordingly.

// COMMAND ----------

implicit val ordering : Ordering[NodeLabel] = leftRightOrd
val subtreeRDD = countedTrain.repartitionAndSortWithinPartitions(partitioner)

// COMMAND ----------

// MAGIC %md
// MAGIC Now that whole branches of leaves reside inside individual partitions, we can ensure that every worker can 
// MAGIC merge all of its data up to the depth of the deepest branch root among all generated branches. This depth is saved to the value `depthLimit` below.

// COMMAND ----------

val depthLimit = partitioner.maxSubtreeDepth

// COMMAND ----------

// MAGIC %md
// MAGIC The last parameter the user have to set is `countLimit`. The value acts as a smoothing parameter for our histogram and a smaller limit gives a more refined histogram. This histogram will act as the end of the path of increasingly refined histograms in which we will search for a final estimate in **stage 4**. Furthermore, a larger limit affects the memory usage of the library, and is currently acting as the
// MAGIC limiting factor in **stage 4**. There are possible solutions that can be applied here and implementing this would be the next step to take for developing the library.
// MAGIC
// MAGIC We note once again that no input leaf is allowed to have a count exceeding the count limit. **getCountLimit** takes as input the RDD of leaves and a  minimum count limit we want,
// MAGIC  and if we find some maximum leaf
// MAGIC  containing a count exceeding the minimum, we return that as a count limit instead. The returned count limit is guaranteed to work in the merging step.

// COMMAND ----------

val minimumCountLimit : Int = 400
val countLimit = getCountLimit(countedTrain, minimumCountLimit)

// COMMAND ----------

// MAGIC %md
// MAGIC We finally merge the leaves and create a `Histogram` class object from the newly constructed cells. `Histogram` class objects contain the original root box `tree`, the total number of points `totalCount`
// MAGIC and a object called `counts` of the class `LeafMap`. A `LeafMap` is the library's abstraction of maps from a set of leaves `NodeLabel`.

// COMMAND ----------

val finestHistogram : Histogram = mergeLeavesHistogram(tree, subtreeRDD, countLimit, depthLimit)
countedTrain.unpersist()

// COMMAND ----------

// MAGIC %md
// MAGIC The **final stage** is now upon us. At this point, there are only two parameters left that the user must set. Remember that we do an adaptive search along a path of increasingly refined histograms.
// MAGIC In every iteration we choose \\(k\\) points evenly distributed along a range of histograms on this path and find the minimum distance estimate within the set of chosen histograms.
// MAGIC The user simply has to choose the value of \\(k\\). In the previous **Theorem**, we saw how \\(k = \big|\Theta\big|\\) affects the statistical performance. We cannot choose to large a \\(k\\) since a
// MAGIC part of **stage 4** scales quadratically in computational cost in relation to the constant. We set `kInMDE` to 10.

// COMMAND ----------

val kInMDE = 10

// COMMAND ----------

// MAGIC %md
// MAGIC The user must also set `numCores` to the number of **cores** used in the cluster. **stage 4** needs this value set as it is the number of partitions it will use internally after a
// MAGIC reduceByKey call in order to achieve good performance.

// COMMAND ----------

val numCores = 4

// COMMAND ----------

// MAGIC %md
// MAGIC We apply the adaptive search and is returned the final estimate in the form of a `Histogram` class object

// COMMAND ----------

val minimumDistanceHistogram : Histogram = getMDE(finestHistogram, countedValidation, validationSize, kInMDE, numCores, true)

// COMMAND ----------

// MAGIC %md
// MAGIC We can turn it into a `DensityHistogram` class object by using the `toDensityHistogram` function. The `DensityHistogram` class values are very similar to `Histogram` class values;
// MAGIC The object stores the final cells on which the density takes non-zero values. Furthermore, a `LeafMap` object maps the cells to tuples of the form (density, volume), i.e. 
// MAGIC for every cell we have stored in the map the value that the density takes on any point wihtin the cell, and the cell's volume.

// COMMAND ----------

val minimumDistanceDensity : DensityHistogram = toDensityHistogram(minimumDistanceHistogram)

// COMMAND ----------

// MAGIC %md
// MAGIC The last thing to do is to normalize the `toDensityHistogram` return value, in order to get a density that integrates to 1 over the whole space. `density` is the final actual density estimate.

// COMMAND ----------

val density : DensityHistogram = minimumDistanceDensity.normalize

// COMMAND ----------

// MAGIC %md
// MAGIC Lets save some information and our `Histogram` to disk (please give a valid directory below).

// COMMAND ----------

/* TODO: write a folder to save data to */
val pathToDataFolder = "dbfs:/users/sandstedt/notebooks"

// COMMAND ----------

val rootPath = s"${pathToDataFolder}/introduction"
val treePath = rootPath + "spatialTree"
val finestResDepthPath = rootPath + "finestRes" 
val mdeHistPath = rootPath + "mdeHist"
 
Vector(tree.rootCell.low, tree.rootCell.high).toIterable.toSeq.toDS.write.mode("overwrite").parquet(treePath) 
Array(finestResDepth).toIterable.toSeq.toDS.write.mode("overwrite").parquet(finestResDepthPath)
minimumDistanceHistogram.counts.toIterable.toSeq.toDS.write.mode("overwrite").parquet(mdeHistPath)
countedValidation.unpersist()
