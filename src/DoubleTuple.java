
/**
 * Class created to hold the values used when 
 * splitting the numeric data.
 * 
 * @author Ashwin
 *
 */

public class DoubleTuple {
	
	public double splitVal;
	public double classVal;
	public double weight;
	
	/**
	 * Constructor for setting the split value, its associated class
	 * value and its corresponding weight
	 * @param sVal
	 * @param cVal
	 * @param wt
	 */
	DoubleTuple(double sVal, double cVal, double wt){
		this.splitVal = sVal;
		this.classVal = cVal;
		this.weight = wt;
	}

}
