import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
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

public class main_f{
	public static void main(String[] args) throws Exception{
		//NaiveBayesSimple nb = new NaiveBayesSimple();
		
//		BufferedReader br_train = new BufferedReader(new FileReader("src/train.arff.txt"));
//		String s = null;
//		long st_time = System.currentTimeMillis();
//		Instances inst_train = new Instances(br_train);
//		System.out.println(inst_train.numAttributes());
//		inst_train.setClassIndex(inst_train.numAttributes()-1);
//		System.out.println("train time"+(System.currentTimeMillis()-st_time));
		//NaiveBayes nb1 = new NaiveBayes();
		//nb1.buildClassifier(inst_train);
		//br_train.close();
		long st_time = System.currentTimeMillis();
		st_time = System.currentTimeMillis();
		
		Classifier classifier = (Classifier) SerializationHelper.read("NaiveBayes.model");
		
		
//		BufferedReader br_test = new BufferedReader(new FileReader("src/test.arff.txt"));
//		Instances inst_test = new Instances(br_test);
//		inst_test.setClassIndex(inst_test.numAttributes()-1);
//		System.out.println("test time"+(System.currentTimeMillis()-st_time));
//		
		

		ArffLoader testLoader = new ArffLoader();
		testLoader.setSource(new File("src/test.arff"));
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