Weighted-Majority
=================

Weighted Majority Trainer

Java Version: 1.7 <br>
Dependencies: WEKA (jar file)<br>

<h5>Prerequisites</h5>
<ul>
<li>The test set and training set should be present in arff format.</li>
<li>WEKA jar file should be linked or referenced.</li>
<li>Class attribute should be the last attribute in the test/training set</li>
</ul>

<p>Implements a weighted majority trainer algorithm. Uses WEKA for parsing through the dataset. The weighted majority implementation used the attributes of the dataset as predictors. Hence there may be a lower performance when compared to WM implementations that use different classifiers as the predictors. This method tolerates the values missing at random. This implementation calculates and prints the Accuracy, G Mean, Cohen's Kappa of the test set.</p>
