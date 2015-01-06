import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Trains a weighted majority classifier using the attributes
 * of the dataset as the predictors. Outputs the accuracy of 
 * the test set, G Mean, Cohen'Kappa and Error rate to the console.
 *  
 * @author Ashwin Karthi
 *
 */

public class WM {
	
	private static Instances dataSet;
	private static ArrayList<Integer> foldMap = new ArrayList<Integer>();
	public static AttributeData[] attributesData;
	private static double accuracy = 0;
	
	/**
	 * Trains the weight for the trainingSet
	 * @param trainingSet
	 */
	private static void trainWeights(Instances trainingSet){
		int i = 0;
		
		//for each attribute train the weight
		for(i=0;i<trainingSet.numAttributes()-1;i++){
			trainAttributeWeight(i,trainingSet);
		}
	}
	
	/**
	 * Finds the accuracy obtained if this particular attribute is selected 
	 * and modifies the associated weight
	 * @param attIndex
	 * @param trainingSet
	 */
	private static void trainAttributeWeight(int attIndex, Instances trainingSet){
		
		if(trainingSet.attribute(attIndex).isNominal()){
			trainNominalAttributeWeight(attIndex, trainingSet,attributesData[attIndex]);
		}
		else{
			trainNumericAttributeWeight(attIndex, trainingSet, attributesData[attIndex]);
		}
	}
	
	/**
	 * Trains the nominal attribute weight based on the number of +ive and -ive
	 * instances for each value of attribute and calculating the approximate accuracy.
	 * Modifies the weight array index corresponding to the attribute
	 * @param attributeIndex
	 * @param trainingSet
	 */
	private static void trainNominalAttributeWeight(int attributeIndex, Instances trainingSet, AttributeData attributeData){
		Instance instance;
		Attribute attribute = trainingSet.attribute(attributeIndex);
		int i = 0, j = 0, k = 0, cnt = 0;
		int limit1 = trainingSet.numInstances();
		int limit2 = attribute.numValues();
		int limit3 = trainingSet.classAttribute().numValues();
		String attvalue = null;
		
		//for each attribute value count the no of instances for each class
		int attValPred[][]  = new int[limit2][limit3];
		attributeData.setNominalAttributeValueClassArray(limit2);
		
		//for each attribute value count the no of instances for each class
		for(i=0;i<limit1;i++){
			
			instance = trainingSet.instance(i);
			attvalue =  instance.stringValue(attribute);
			
			for(j=0;j<limit2;j++){
				if(attvalue.equals(attribute.value(j))){
					attValPred[j][(int)instance.classValue()] ++;
				}
			}
		}
		
		//Set the class to be predicted for each value of the attribute
		for(j=0;j<limit2;j++){
			int maxPredCnt = attValPred[j][0];
			int maxPredCntClassIndex = 0;
			cnt = attValPred[j][0];
			
			for(k=1;k<limit3;k++){
				cnt += attValPred[j][k];
				if(attValPred[j][k] > maxPredCnt){
					maxPredCnt           = attValPred[j][k];
					maxPredCntClassIndex = k; 
				}
			}
			
			if(cnt == 0){
				attributeData.attValClass[j] = new Random(7).nextInt()%limit3;
				attributeData.weight[j] = 0;
				continue;
			}
			
			//set the class value for each value of nominal attribute
			attributeData.attValClass[j] = maxPredCntClassIndex;
			
			//set the prediction weight for each value of nominal attribute
			attributeData.weight[j] = (double)((double)maxPredCnt)/((double)cnt);
			
		}
		
		attributeData.discard = false;
		attributeData.numeric = false;
		
		/*System.out.println("\nAttribute = "+attribute.name());
		for(j=0;j<limit2;j++){
			System.out.println("\n\tAttVal = "+attribute.value(j)+"\tPred Class = "+attributeData.attValClass[j]);
			for(k=0; k<limit3; k++){
				System.out.print("\t"+k+"th-class = "+attValPred[j][k]);
			}
			System.out.print("\tweight = "+attributeData.weight[j]);
		}*/
	}
	
	/**
	 * Trains the numeric attribute and updates the weight in the corresponding attributeData
	 * @param attributeIndex
	 * @param trainingSet
	 * @param attributeData
	 */
	private static void trainNumericAttributeWeight(int attributeIndex, Instances trainingSet, AttributeData attributeData){
		
		Instance instance;
		
		ArrayList<Double> splits = new ArrayList<Double>();
		ArrayList<ArrayList<Integer>> splitData = new ArrayList<ArrayList<Integer>>();
		ArrayList<Integer> singleSplit;
		ArrayList<Integer> splitMaxClassCnt = new ArrayList<Integer>();
		ArrayList<Integer> splitTotalInstCnt = new ArrayList<Integer>();
		ArrayList<DoubleTuple> listSplitClassWt = new ArrayList<DoubleTuple>();
		
		int i = 0, j = 0;
		int numInsts = trainingSet.numInstances();				    //no of instances
		int numClasses = trainingSet.classAttribute().numValues();	//no of classes
		int classIndex1 = 0;
		int max1 = 0, total1 = 0;
		int []totalCnt = new int[numClasses];
		
		double pval = 0;
		double curval = 0;
		double curclass;										    //weight is the accuracy
		
		trainingSet.sort(attributeIndex);
		instance = trainingSet.instance(0);
		
		/*for(i=0;i<numInsts;i++){
			for(j=0;j<trainingSet.numAttributes()-1;j++){
				System.out.print(trainingSet.instance(i).value(j)+"\t");
			}
			System.out.println();
		}*/
		
		pval = trainingSet.instance(0).value(attributeIndex);
		curval = trainingSet.instance(0).value(attributeIndex);
		
		for(i = 0; i<numInsts; i++){
			
			instance = trainingSet.instance(i);
			curval = instance.value(attributeIndex);
			curclass = instance.classValue();
			
			if(curval != pval){
				
				//value changes add split to the array list
				/*System.out.println("\nAddidn split "+pval+"\t"+curval);*/
				splits.add((pval+curval)/2);
				
				//push the values in totalCnt to a new arrayList in splitData
				//that maintains the information corresponding to the split
				singleSplit = new ArrayList<Integer>();
				
				for(j=0; j<numClasses; j++){
					
					//add it to the array list
					singleSplit.add(totalCnt[j]);
					
					//set the class cnt to 0 for the next split
					totalCnt[j] = 0;
				}
				
				//now add this to the splitData
				splitData.add(singleSplit);
				
				pval = curval;
			}
			
			totalCnt[(int)curclass]++;
		}
		
		//add for final value of attribute
		singleSplit = new ArrayList<Integer>();
		for(j=0; j<numClasses; j++){
			singleSplit.add(totalCnt[j]);
		}
		
		// now add this to the splitData
		splitData.add(singleSplit);
		splits.add(curval+1);
		
		// set the appropriate values in attributeData
		attributeData.numeric = true;
		
		/*System.out.println("Splits size = "+splits.size());
		System.out.println("Splits data size = "+splitData.size());*/
		
		// for each split assign the weight
		for(i=0; i<splits.size(); i++){
			
			//find the class that repeats the most for each split
			//keep a look whether the next split has the same class
			
			//obtain the array list that contains the sum of each classes
			//for split 'i'
			singleSplit = splitData.get(i);
			
			//to get the split value splits.get(i);
			
			//for each split find the max class and assign it to that split
			
			max1 = singleSplit.get(0);
			classIndex1 = 0;
			total1 = max1;
			
			/*System.out.println("\nPrinting the single split for "+splits.get(i));
			for(j=0;j<singleSplit.size(); j++){
				System.out.print(singleSplit.get(j)+"\t");
			}*/
			
			for(j = 1; j<singleSplit.size(); j++){
				
				total1 += singleSplit.get(j);
				
				if(max1 < singleSplit.get(j)){
					max1 = singleSplit.get(j);
					classIndex1 = j;
				}
			}
			
			// now classIndex1 contains the class predicted and max1 contains the count of instances
			// that have class value pointed by classIndex1
			listSplitClassWt.add(new DoubleTuple(splits.get(i),classIndex1,((double)(double)max1/(double)total1)));
			
			/*System.out.print(" **** Split =  "+listSplitClassWt.get(i).splitVal);
			System.out.print("\tClass =  "+listSplitClassWt.get(i).classVal);
			System.out.print("\tWt =  "+listSplitClassWt.get(i).weight+"\n");*/
			
			//record the cnt of instances that belonged to the class predicted
			splitMaxClassCnt.add(max1);
			
			//record the total instances associated with this split
			splitTotalInstCnt.add(total1);
		}
		
		// can combine the splits that point to the same value
		
		
		pval = listSplitClassWt.get(0).classVal;
		total1 = splitTotalInstCnt.get(0);
		max1 = splitMaxClassCnt.get(0);
				
		for(i=1; i<listSplitClassWt.size(); i++){
			
			if(listSplitClassWt.get(i).classVal != pval){
				
				Double psplit =  listSplitClassWt.get(i-1).splitVal;
				//insert into attribute listSplitClassWt
				attributeData.listSplitClassWt.add(new DoubleTuple(psplit,pval,((double)(double)max1/(double)total1)));
				
				pval = listSplitClassWt.get(i).classVal;
				total1 = 0;
				max1 = 0;
			}
			
			total1 += splitTotalInstCnt.get(i);
			max1 += splitMaxClassCnt.get(i);
		}
		
		//for final split
		Double psplit =  listSplitClassWt.get(i-1).splitVal;
		attributeData.listSplitClassWt.add(new DoubleTuple(psplit,pval,((double)(double)max1/(double)total1)));
		
		/*System.out.println("\n****** for attribute = "+trainingSet.attribute(attributeIndex).name());
		//display the split weight and class
		for(i=0; i<attributeData.listSplitClassWt.size(); i++){
			System.out.print("Split =  "+attributeData.listSplitClassWt.get(i).splitVal);
			System.out.print("\tClass =  "+attributeData.listSplitClassWt.get(i).classVal);
			System.out.print("\tWt =  "+attributeData.listSplitClassWt.get(i).weight+"\n");
		}*/
	}
	
	/**
	 * Predicts the class of the instance which belongs to the testSet
	 * @param testSet
	 * @param instance
	 * @return The class index which was predicted by the weighted majority algorithm
	 */
	private static double getClass(Instances testSet, Instance instance){
		double predClass = 0;
		int i = 0, j = 0, cnt = 0;
		double []classWt = new double[testSet.classAttribute().numValues()];
		Attribute att;
		
		for(i=0;i<instance.numAttributes()-1;i++){
			
			if(attributesData[i].discard == true)
				continue;
			
			att = testSet.attribute(i);
			
			if(attributesData[i].numeric){
				
				boolean rangeFound = false;
				
				for(j=0; j<attributesData[i].listSplitClassWt.size(); j++){
					
					if(attributesData[i].listSplitClassWt.get(j).splitVal > instance.value(i)){
						int index = (int) attributesData[i].listSplitClassWt.get(j).classVal;
						if(index < 0){
							cnt++;
							continue;
						}
						classWt[index] += attributesData[i].listSplitClassWt.get(j).weight;
						rangeFound = true;
						break;
					}
					
				}
				
				// when value is out of bounds
				if(rangeFound == false){ 
					int index = (int) attributesData[i].listSplitClassWt.get(j-1).classVal;
					if(index < 0){
						cnt++;
						continue;
					}
					classWt[index] += attributesData[i].listSplitClassWt.get(j-1).weight;
				}
				
			}
			
			else{
				for(j=0; j<attributesData[i].noOfNomValues; j++){
					if(att.value(j).equals(instance.stringValue(i))){
						
						int index = attributesData[i].attValClass[j];
						if(index < 0){
							cnt++;
							continue;
						}
						classWt[index]	 += attributesData[i].weight[j];
						break;
					}
				}
			}
		}
		
		double maxWt = 0;
		for(i=0;i<instance.classAttribute().numValues();i++){
			if(classWt[i] > maxWt){
				predClass = i;
				maxWt = classWt[i]; 
			}
		}
		//System.out.println("\nNeg = "+cnt);
		return predClass;
	}
	
	/**
	 * Calculates the accuracy of the given test set and puts the value in variable accuracy
	 * @param testSet
	 */
	public static void getAccuracy(Instances testSet){
		int i=0;
		int correctPred = 0;
		double predClass = 0, actualClass = 0;
		double confusionMatrix[][] = new double[testSet.classAttribute().numValues()+1][testSet.classAttribute().numValues()+1]; 
		
		for(i=0; i<testSet.numInstances(); i++){
			
			predClass = getClass(testSet,testSet.instance(i));
			actualClass = testSet.instance(i).classValue();
			
			//System.out.println("\nPredicted Class =" + predClass+"\tActual = "+actualClass);
			
			if(predClass == actualClass)
				correctPred ++;
			confusionMatrix[(int)actualClass][(int)predClass]++;
		}
		accuracy = (double)correctPred/(double)(testSet.numInstances());
		System.out.println("\nAccuracy = "+accuracy);
		System.out.println("\nError = "+(1-accuracy));
		
		for(i=0;i<confusionMatrix.length;i++){
			for(int j=0;j<confusionMatrix.length;j++){
				if(j!=confusionMatrix.length-1)
					confusionMatrix[i][confusionMatrix.length-1] += confusionMatrix[i][j];
				if(i!=confusionMatrix.length-1)
					confusionMatrix[confusionMatrix.length-1][j] += confusionMatrix[i][j];
			}
		}
		
		confusionMatrix[confusionMatrix.length-1][confusionMatrix.length-1] /= 2;
		
		System.out.println("\nCohen's Kappa Value = "+calculateKappa(confusionMatrix));
		System.out.println("\nGMean Value = "+calculateGMean(confusionMatrix));
	}
	
    /**
     * Calculates the Cohen's Kappa value for the given Confusion Matrix
     * @param testSetConfusionMatrix
     * @return The Cohen Kappa value
     */
    public static double calculateKappa(double testSetConfusionMatrix[][])
    {
    	/*for(int i = 0; i<testSetConfusionMatrix.length;i++){
    		for(int j=0; j<testSetConfusionMatrix[0].length;j++)
    			System.out.print(testSetConfusionMatrix[i][j]+"\t");
    		System.out.println();
    	}*/
    	
        int classCount = testSetConfusionMatrix.length-1;
        
        double kappa=0;
        
        double dii=0;
        
        for(int i=0;i < classCount;i++)
        {
            dii += testSetConfusionMatrix[i][i];
        }
        
        double Tmul=0;
        
        for(int i=0;i<classCount;i++)
        {
            
            Tmul += testSetConfusionMatrix[classCount][i]*testSetConfusionMatrix[i][classCount];
            
        }
        
//        System.out.println("\ndii ="+dii);
//        System.out.println("\ntmul ="+Tmul);
//        System.out.println("\nnum ="+(testSetConfusionMatrix[classCount][classCount]*dii-Tmul));
        
        kappa = (double)(testSetConfusionMatrix[classCount][classCount]*dii-Tmul)
        		/(double)(testSetConfusionMatrix[classCount][classCount]
        				*testSetConfusionMatrix[classCount][classCount]-Tmul);
        
        return kappa;
    }
    
    /**
     * Calculates the GMean value for the given Confusion Matrix
     * @param testSetConfusionMatrix
     * @return The GMean Value
     */
    public static double calculateGMean(double testSetConfusionMatrix[][])
    {
    	
        int classCount=testSetConfusionMatrix.length-1;
        double gMean=1;
        
        //Find and multiply all the senstivities
        for(int i=0;i<classCount;i++){
            
        	gMean=gMean*(1+testSetConfusionMatrix[i][i])/(1+testSetConfusionMatrix[i][classCount]);
        }
        
        //classCount root
        gMean=Math.pow(gMean,((double)((double)1.00)/((double)classCount)));
        
        return gMean;
    }
	
	/**
	 * Generates a mapping for each instance with the associated fold.
	 * Produces foldMap[i] = foldValue where 'i' is the i-th instance in the dataSet
	 * and foldValue is the fold to which it is associated to.
	 * @param folds 
	 */
	public static void foldInstances(int folds){
		
		int i = 0, j = 0, positiveCount = 0, negativeCount = 0;
		int pinfold = 0, ninfold = 0;
		int remainingPositive = 0, remainingNegative = 0;
		
		ArrayList<Integer> pmap = new ArrayList<Integer>();
		ArrayList<Integer> nmap = new ArrayList<Integer>();
		
		//count the number of positive and negative instances
		for(i=0; i<dataSet.numInstances(); i++){	
			if(dataSet.instance(i).classValue()==0)
				negativeCount++;
			else
				positiveCount++;
		}
		
		pinfold = positiveCount/folds;
		remainingPositive = positiveCount%folds;
		
		ninfold = negativeCount/folds;
		remainingNegative = negativeCount%folds;
		
		for(i=0;i<folds;i++){
			for(j=0;j<pinfold;j++){
				pmap.add(i);
			}
			for(j=0;j<ninfold;j++){
				nmap.add(i);
			}
		}
		
		//evenly distribute the remaining positives into the map 
		if(remainingPositive!=0){
			for(j=0;j<remainingPositive;j++)
				pmap.add(j);
		}
		
		//evenly distribute the remaining negatives into the map
		if(remainingNegative!=0){
			for(j=0;j<remainingNegative;j++)
				nmap.add(j);
		}
		
		//now schuffle pmap and nmap
		Collections.shuffle(pmap);
		Collections.shuffle(nmap);
		
		for(i=0;i<dataSet.numInstances();i++){
			Instance instance = dataSet.instance(i);
			if(instance.classValue() == 0){
				//get the fold number from nmap and add it foldsMap
				foldMap.add((int)nmap.get(0));
				nmap.remove(0);
			}
			else{
				//get the fold number from pmap and add it to foldsMap
				foldMap.add((int)pmap.get(0));
				pmap.remove(0);
			}
		}
		
	}
	
	/**
	 * Main method of the class. Sets the trainingSet and the testSet based on
	 * the command line arguments passed to the program.
	 * @param args
	 */
	public static void main(String args[]){
		
		if(args.length!=2){
			System.out.println("Usage: java -jar filename.jar trainingSetFile.arff testSetFile.arff");
			return;
		}
		
		BufferedReader reader;
		Instances testSet = null;
		
		try{	
			//load the Instances using weka tool
			reader = new BufferedReader(new FileReader(args[0]));
			dataSet = new Instances(reader);
			reader.close();
		}
		catch(Exception e){
			System.out.print("\nError thrown in main function1: "+e);
		}
		
		try{	
			//load the Instances using weka tool
			reader = new BufferedReader(new FileReader(args[1]));
			testSet = new Instances(reader);
			reader.close();
		}
		catch(Exception e){
			System.out.print("\nError thrown in main function2: "+e);
		}
		
		dataSet.setClassIndex(dataSet.numAttributes() - 1);
		attributesData = new AttributeData[dataSet.numAttributes()-1];
		
		testSet.setClassIndex(testSet.numAttributes() - 1);
		attributesData = new AttributeData[dataSet.numAttributes()-1];
		
		//create a 10 fold map for the dataset
		//foldInstances(10);
		
		for(int i=0;i<attributesData.length;i++){
			attributesData[i] = new AttributeData();
		}
		
		trainWeights(dataSet);
		getAccuracy(testSet);
		
	}

}
