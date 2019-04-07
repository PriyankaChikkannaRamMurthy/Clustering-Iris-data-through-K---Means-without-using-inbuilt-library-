import numpy as np

def reassign_cluster(clus_arr, k, initial_data):
    for n in range(0, k):
        index_val = np.where(clus_arr == n)
        cluster_same = x_train[index_val]
        new_center = []
        if (np.shape(cluster_same)[0] != 0):
            same_cluster = np.sum(cluster_same, axis=0)
            size_of = np.shape(cluster_same)[0]
            new_center = same_cluster / size_of
            initial_data[n] = new_center
    #print("New Centroid data : ", initial_data)
    #print(type(initial_data))
    return initial_data

def centroid_distance(x_train,initial_data):
    clusters = []
    for point in x_train:
        cluster_distance = []
        for centroid in initial_data:
            #print(point, " - ", centroid)
            value = np.subtract(point,centroid)
            #distance = value(x_train) - value(initial_data)
            #print(distance)
            #print(value)
            squared_value = np.square(value)
            #print(squared_value)
            sum_squared = np.sum(squared_value)
            #print(sum_squared)
            sqrt_val = np.sqrt(sum_squared)
            #print(sqrt_val)
            cluster_distance.append(sqrt_val)
        clusters.append(cluster_distance.index(min(cluster_distance)))
    print("Cluster Labels :" , clusters)
    return clusters

file = "iris_data.csv"
x_features = np.loadtxt(file, delimiter=',', usecols=[0, 1, 2, 3])
#print(x_features)
#print(x_features.shape)

y_label  = np.loadtxt(file, delimiter=',', usecols=[4], dtype=np.str)
y_data = y_label.reshape((150,1))
#print(y_data)
#print(y_data.shape)

for j in range(0,150):
    if y_data[j] == 'Iris-setosa':
        y_data[j]= 0
    elif y_data[j] == 'Iris-versicolor':
        y_data[j]= 1
    elif y_data[j] == 'Iris-virginica':
        y_data[j]= 2
y_data = np.array(y_data, dtype=np.int64)
#print(y_data)
#print(type(y_data))

whole_matrix=np.concatenate((x_features, y_data), 1)
#print(whole_matrix)
#print(type(whole_matrix))
#print(whole_matrix.shape)

np.random.shuffle(whole_matrix)
#print(whole_matrix)
#print(whole_matrix.shape)

x_new =whole_matrix[:, :-1].copy()
#print(x_new)
#print(type(x_new.shape))

y_new = np.array(whole_matrix[:, -1].copy(), dtype=np.int64)
y_new_label = y_new.reshape((150,1))
#print(y_new_label)
#print(y_new_label.shape)

x_train = x_new[:120,:].copy()
#print(x_train)
#print(x_train.shape)

y_train = y_new_label[:120,:].copy()
#print(y_train)
#print(y_train.shape)

#initial_data = np.random.rand(3,4)
initial_data = np.random.uniform(low=0.1, high=8.0, size=(3,4))
#print("Initial Cluster data: ", initial_data)
#print(initial_data.shape)


k = int(input("Enter the number of clusters you want the dataset to be classified accordingly: "))


for i in range(0,100):
    print("Centroid data :")
    print(initial_data)
    cluster_new = centroid_distance(x_train, initial_data)
    clus_arr = np.asarray(cluster_new)
    reassign_val = reassign_cluster(clus_arr, k, initial_data)
    print(reassign_val)






