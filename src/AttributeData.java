import java.util.ArrayList;

/**
 * Class created to hold the properties of the 
 * attribute when it acts as the predictor 
 * 
 * @author Ashwin
 *
 */

public class AttributeData {

	boolean discard;
	
	// Numeric attribute related field
	
	boolean numeric;
	ArrayList<DoubleTuple> listSplitClassWt = new ArrayList<DoubleTuple>();
	
	
	//Nominal Attribute related data fields and methods
	
	int 	attValClass[];			    //points to the index of the class for each nominal value
	double  weight[];					//the weight assiociated with the class prediction of each nominal value
	int 	noOfNomValues;				//the number of nominal values of the nominal attribute
	
	void setNominalAttributeValueClassArray(int noOfNomValues){
		this.noOfNomValues = noOfNomValues;
		attValClass = new int[noOfNomValues];
		weight = new double[noOfNomValues];
	}
}
