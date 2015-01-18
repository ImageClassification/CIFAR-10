CIFAR-10
========

This repository will host the various image classification techniques used in experimentations.

After various experimentation as discussed above, we develop an ensemble learning system that uses the best performing methods that we found experimentally. Primarily we use the results from various K-Means with L2SVM parameter variations and Gist with SVM, combining it with a mixture of moderately performing classifiers such as Random Forest, Kernel Multinomial Logistic Regression.
The ensemble system uses a biased voting strategy where the most common class label predicted by each classifier is considered as the final predicted class label. However, if there is a tie, we default to the label predicted by the strongest individual classifier.
After using this ensemble classifier, we observer dramatic improvements in performance. The best combination gave a classification accuracy of 0.5965 on the test data set.

For more details please refer to the report "bayseians_report.pdf" 
