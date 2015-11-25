import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics

val trainData = sc.textFile("/Users/wesleyhoffman/Documents/EECS738/FinalProject/EECS738_Train.csv")
val parsedTrainData = trainData.map { line =>
  val parts = line.split(',').map(_.toDouble)
  LabeledPoint(parts(1), Vectors.dense(parts.drop(2)))
}

//Split the data into training and test sets (30% held out for testing)
val splits = parsedTrainData.randomSplit(Array(0.7, 0.3))
val (trainingData, testData) = (splits(0), splits(1))

//Train RandomForest Model
val numClasses = 2
val categoricalFeaturesInfo = Map[Int, Int]()
val numTrees = 12
val featurSubsetStrategy = "auto"
val impurity = "gini"
val maxDepth = 7
val maxBins = 32

val model = RandomForest.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo,
  numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)

val labelAndPreds = testData.map { point =>
  val prediction = model.predict(point.features)
  (point.label, prediction)
}

val metrics = new BinaryClassificationMetrics(labelAndPreds)
