import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors

val trainData = sc.textFile("/Users/wesleyhoffman/Documents/EECS738/Machine_Learning_Final_Project/SparkProject/EECS738_Train.csv")
val parsedTrainData = trainData.map { line =>
  val parts = line.split(',').map(_.toDouble)
  LabeledPoint(parts(1), Vectors.dense(parts.drop(2)))
}
     sdfds
//Split the data into training and test sets (20% held out for testing)
val splits = parsedTrainData.randomSplit(Array(0.7, 0.3))
val (trainingData, testData) = (splits(0), splits(1))

//Train RandomForest Model
val numClasses = 2
val categoricalFeaturesInfo = Map[Int, Int]()
val numTrees = 15
val featureSubsetStrategy = "auto"
val impurity = "gini"
val maxDepth = 7
val maxBins = 100

val model = RandomForest.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo,
  numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)

val labelAndPreds = testData.map { point =>
  val prediction = model.predict(point.features)
  (point.label, prediction)
}

val metrics = new BinaryClassificationMetrics(labelAndPreds)
val auc = metrics.areaUnderROC()

// Precision by threshold
val precision = metrics.precisionByThreshold
precision.foreach { case (t, p) =>
    println(s"Threshold: $t, Precision: $p")
}

// Recall by threshold
val recall = metrics.recallByThreshold
recall.foreach { case (t, r) =>
    println(s"Threshold: $t, Recall: $r")
}

println("AUC = " + auc)

val testErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / testData.count()
println("Test Error = " + testErr)
//println("Learned classification forest model:\n" + model.toDebugString)
