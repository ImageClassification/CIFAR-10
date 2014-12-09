import java.io.File;
import java.util.ArrayList;
import java.util.Enumeration;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ArffLoader;
import weka.core.converters.CSVSaver;
import weka.core.converters.Loader;


public class RandomForestImageClassifier 
{
	
	public static void main(String[] args) throws Exception 
	{
		Classifier classifier = (Classifier) SerializationHelper.read("classifier500.model");
		
		ArffLoader testLoader = new ArffLoader();
		testLoader.setSource(new File("test.arff"));
		testLoader.setRetrieval(Loader.BATCH);
		Instances testDataSet = testLoader.getDataSet();


		Attribute testAttribute = testDataSet.attribute("class");
		testDataSet.setClass(testAttribute);

		int correct = 0;
		int incorrect = 0;
		FastVector attInfo = new FastVector();
		attInfo.addElement(new Attribute("Id"));
		attInfo.addElement(new Attribute("Category"));
	
		Instances outputInstances = new Instances("predict",attInfo,testDataSet.numInstances());
		
		Enumeration testInstances = testDataSet.enumerateInstances();
		int index  = 1;
		while (testInstances.hasMoreElements()) {
			Instance instance = (Instance) testInstances.nextElement();
			double classification = classifier.classifyInstance(instance);
			/*if (((int)instance.classValue()-(int)classification) == 0)
				correct++;
			else
				incorrect++;*/
			Instance predictInstance = new Instance(outputInstances.numAttributes());
			predictInstance.setValue(0, index++);
			predictInstance.setValue(1, (int)classification + 1);
			outputInstances.add(predictInstance);
		}

		
		System.out.println("Correct Instance: "+correct);
		System.out.println("IncCorrect Instance: "+incorrect);
		double accuracy = (double)(correct)/(double)(correct+incorrect);
		System.out.println("Accuracy: "+accuracy);
		CSVSaver predictedCsvSaver = new CSVSaver();
		predictedCsvSaver.setFile(new File("predict.csv"));
		predictedCsvSaver.setInstances(outputInstances);
		predictedCsvSaver.writeBatch();

		System.out.println("Prediciton saved to predict.csv");
	}

}
