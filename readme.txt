1A.py
	A) The question include 2 Plots. One for RMSE vs epochs for training set and other for RMSE vs epochs for validation set.
	B) The optimal parameter was founded using the normal equation. Then RMSE for both testing and training set was reported for all 5 folds.

1B.py
	A) L1 and L2 was found out using GridsearchCV library form sci-kit library.
		Also the RMSE vs iterations curve was plotted for both L1 and L2 on the Tets set.
1C.py
	A) First plot includes the scatter plot along with the best fit line using linear regression without regularisation.
	B) Second plot includes L2 regularisation VS data points on the new best fit line.
	C) Third plot includes L1 regularisation VS data points on the new best fit line.

2A.py
	A) Logistic regression was implemented from scratch and L1, L2 were found out using Sci-kit learn library. The accuracy and RMSE were reported and plotted for validation, test, 			and train set for both using L1 and L2.
	B) L1 and L2 were found out for the logistic regression. Then the accuracy of each 10 classes were reported using one-vs-rest approach. This was done for both on test and train set 
		using both L1 and L2.
	C) Reciever Operating Curve for the classes 0 to 9 is plotted separately.
