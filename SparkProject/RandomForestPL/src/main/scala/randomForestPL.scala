import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.classification.RandomForestClassificationModel
import org.apache.spark.ml.feature.{StringIndexer, IndexToString, VectorIndexer}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors

object RndmForst{
  def main(args: Array[String]){
    val logFile = "/Users/wesleyhoffman/Documents/EECS738/Machine_Learning_Final_Project/SparkProject/RFPL.md"
    val conf = new SparkConf().setAppName("Random Forest with ML Pipeline")
    val context = new SparkContext(conf)

    // Load and parse the data file, converting it to a DataFrame.
    val trainData = context.textFile("/Users/wesleyhoffman/Documents/EECS738/Machine_Learning_Final_Project/SparkProject/EECS738_Train.csv")
    val parsedTrainData = trainData.map { line =>
      val parts = line.split(',').map(_.toDouble)
      LabeledPoint(parts(1), Vectors.dense(parts.drop(2)))
    }
    val trainDataDF = parsedTrainData.toDF();

    val paramGrid = new ParamGridBuilder().build()

    // // Index labels, adding metadata to the label column.
    // // Fit on whole dataset to include all labels in index.
    // val labelIndexer = new StringIndexer()
    //   .setInputCol("label")
    //   .setOutputCol("indexedLabel")
    //   .fit(trainDataDF)
    // // Automatically identify categorical features, and index them.
    // // Set maxCategories so features with > 4 distinct values are treated as continuous.
    // val featureIndexer = new VectorIndexer()
    //   .setInputCol("features")
    //   .setOutputCol("indexedFeatures")
    //   .setMaxCategories(4)
    //   .fit(trainDataDF)

    // Split the data into training and test sets (30% held out for testing)
    val Array(trainingData, testData) = trainDataDF.randomSplit(Array(0.7, 0.3))

    // Train a RandomForest model.
    val rf = new RandomForestClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")
      .setNumTrees(10)

    // // Convert indexed labels back to original labels.
    // val labelConverter = new IndexToString()
    //   .setInputCol("prediction")
    //   .setOutputCol("predictedLabel")
    //   .setLabels(labelIndexer.labels)

    // Chain indexers and forest in a Pipeline
    val pipeline = new Pipeline()
      .setStages(Array(rf))

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new BinaryClassificationEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(10) // Use 3+ in practice

    val cvModel = cv.fit(trainingData)

    // Make predictions.
    val predictions = cvModel.transform(testData)

    // val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]
    // println("Learned classification forest model:\n" + rfModel.toDebugString)
  }
}
