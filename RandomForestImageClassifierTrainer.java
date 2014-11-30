import java.io.File;

import weka.classifiers.functions.LibSVM;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ArffLoader;
import weka.core.converters.Loader;
import weka.gui.beans.Classifier;


public class RandomForestImageClassifierTrainer 
{

	public static void main(String args[]) throws Exception
	{
		ArffLoader trainLoader = new ArffLoader();
		trainLoader.setSource(new File("train.arff"));
		trainLoader.setRetrieval(Loader.BATCH);
		Instances trainDataSet = trainLoader.getDataSet();
		Attribute trainAttribute = trainDataSet.attribute("class");
		
		trainDataSet.setClass(trainAttribute);
		//trainDataSet.deleteStringAttributes();
		
	
		RandomForest classifier = new RandomForest();
		classifier.setNumTrees(500);
		classifier.setMaxDepth(30);
		classifier.setDebug(true);
		
		final double startTime = System.currentTimeMillis();
		classifier.buildClassifier(trainDataSet);
		final double endTime = System.currentTimeMillis();
		double executionTime = (endTime - startTime)/(1000.0);
		System.out.println("Total execution time: " + executionTime );

		SerializationHelper.write("classifier500.model", classifier);
		System.out.println("Saved trained model to classifier.model");
	}
}
