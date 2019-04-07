# Clustering-Iris-data-through-K---Means-without-using-Python-inbuilt-library-

File name:
Kmeans_Prg.py

iris_data.csv - to be present in the same folder as the python file is.

value of k [to be given as input to the program]

Code Logic:
 Initially it was coded using the Jupyter Notebook and then downloaded as python file and been submitted.
 
 Pycharm is the IDE used.
 
 Python - 3.6 
 
 Numpy and matplotlib libraries used
 
To be run as:
python Kmeans_Prg.py
 
 Iris data set is downloaded and converted to a csv file and read as the x and y matrices.
 Later using the Euclidean distance formula the values are computed for each x_train datapoint
 Least sum squared values are computed and squareroot of each is calculated.
 At each iteration, the cluster center values are re-estimated using the reassign_cluster() function call and again the x_train datapoints are re-estimated for new centroid point distance and similarly re-grouped for new or older cluster based on the minimum distance to centroid point. 
 The program stops when it finds that the clusters are not re-assigned with the new values. 
 Graph is plotted keeping in mind for the Elbow method of evaluation to analyse the number of clusters required to efficiently classify the iris dataset.


This is the summary of the project code.

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Environment: Python 3.6

IDE used : Pycharm 

Libraries used: Numpy, matplotlib

1. Problem Sections:
• Loading Iris Dataset and reading it
• The Iris dataset is taken from the http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data repository.
• Later it is converted to a csv readable file format. It is read into 2 matrices with “x_features” of 150x4 matrix and “y_data” of 150x1 matrix.

2. Data:
Iris data set where it is downloaded in the local machine : “iris_data.csv”

3. Method:
The entire method call is ran over for 100 iterations of k value to get more accurate cluster groups. centroid_distance(x_train,initial_data): this method used for calculating the Euclidian distance of each x_train data points with the initial centroid points. On iterating over this, we get the list of cluster values which on minimum Euclidian distance to the centroid point is grouped to one cluster and likewise to other clusters. Euclidean distance is computed as stated in the below diagram: Euclidean distance = Square root(sum_Squared_errors) , for each centroid point based on the k value is computed. 
reassign_cluster(clus_arr, k, initial_data): this method is used for reassigning the cluster values and arriving at a new cluster data with new centroid points by calculating the average datapoint of each clusters data i.e., the grouped x_train dataset with already classified & clustered data. Re-estimating the cluster centers by assuming the previous datasets center values to be correct. 



Results:
Output are the Cluster labels after each of the 100 iterations are printed out until it the cluster values continues to remain the same for all the iterations of k value.


Ex Output:
Initial Cluster data: 
[[5.50624642 5.8726267  1.60691047 3.00223574]
 [5.17426242 5.2527822  0.29767993 0.63287611]
 [0.44923311 4.87780927 5.0284284  1.60000142]]
 
Enter the number of clusters you want the dataset to be classified accordingly: 3

Centroid data :
[[5.50624642 5.8726267  1.60691047 3.00223574]
 [5.17426242 5.2527822  0.29767993 0.63287611]
 [0.44923311 4.87780927 5.0284284  1.60000142]]
 
Cluster Labels :
[1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1]

[[6.28648649 2.90540541 4.95945946 1.7027027 ]
 [5.01086957 3.30869565 1.62173913 0.30217391]
 [0.44923311 4.87780927 5.0284284  1.60000142]]
 
Centroid data :
[[6.28648649 2.90540541 4.95945946 1.7027027 ]
 [5.01086957 3.30869565 1.62173913 0.30217391]
 [0.44923311 4.87780927 5.0284284  1.60000142]]
 
Cluster Labels : 
[1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1]

Based on Elbow method of Evaluation for k-means :
 
With the help of Elbow method for clustering analysis of Iris dataset which is used to find the appropriate number of clusters in dataset, we plot the  x-axis for different ‘k’ values and y-axis plotted for sum of all the squared values within each cluster we find that at k=3 all of the iris dataset found to be classified into 3 clusters with the shortest or minimum intra-cluster values.



References:

https://www.geeksforgeeks.org/numpy-append-python/
https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.random.rand.html
https://www.quora.com/How-do-I-take-input-from-the-user-in-Python
https://stackoverflow.com/questions/10625096/extracting-first-n-columns-of-a-numpy-matrix
https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.insert.html
