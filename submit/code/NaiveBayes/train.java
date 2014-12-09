import java.io.File;
import java.text.AttributedCharacterIterator.Attribute;

import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ArffLoader;
import weka.core.converters.Loader;

public class train{
public static void main(String args[]) throws Exception
	{
		ArffLoader trainLoader = new ArffLoader();
		trainLoader.setSource(new File("src/train.arff"));
		trainLoader.setRetrieval(Loader.BATCH);
		Instances trainDataSet = trainLoader.getDataSet();
		weka.core.Attribute trainAttribute = trainDataSet.attribute("class");
		
		trainDataSet.setClass(trainAttribute);
		//trainDataSet.deleteStringAttributes();
		
	
		NaiveBayes classifier = new NaiveBayes();
		
		final double startTime = System.currentTimeMillis();
		classifier.buildClassifier(trainDataSet);
		final double endTime = System.currentTimeMillis();
		double executionTime = (endTime - startTime)/(1000.0);
		System.out.println("Total execution time: " + executionTime );

		SerializationHelper.write("NaiveBayes.model", classifier);
		System.out.println("Saved trained model to classifier.model");
	}
}