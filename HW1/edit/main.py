import os
import matplotlib.pyplot as plt
from LogisticRegression import logistic_regression
from LRM import logistic_regression_multiclass
from DataReader import *

data_dir = "../data/"
train_filename = "training.npz"
test_filename = "test.npz"
    
def visualize_features(X, y):
  '''This function is used to plot a 2-D scatter plot of training features. 

    Args:
        X: An array of shape [n_samples, 2].
        y: An array of shape [n_samples,]. Only contains 1 or -1.

    Returns:
        No return. Save the plot to 'train_features.*' and include it
        in submission.
  '''
  ### YOUR CODE HERE
  plt.clf()  
  p = np.where(y==1)
  n = np.where(y==-1)

  plt.plot(X[p, 0], X[p, 1], 'or', marker='o')
  plt.plot(X[n, 0], X[n, 1], 'ob', marker='x')

  plt.xlim([-1,0])
  plt.ylim([-1,0])

  plt.title("train features")
  plt.xlabel("Symmetry")
  plt.ylabel("Intensity")
  plt.savefig("train_features.png")

  plt.show() 
    ### END YOUR CODE

def visualize_result(X, y, W):
  '''This function is used to plot the sigmoid model after training. 

  Args:
    X: An array of shape [n_samples, 2].
    y: An array of shape [n_samples,]. Only contains 1 or -1.
    W: An array of shape [n_features,].
    
  Returns:
    No return. Save the plot to 'train_result_sigmoid.*' and include it
    in submission.
  '''
  ### YOUR CODE HERE
  plt.clf()  
  plt.plot(X[y==1, 0], X[y==1, 1], 'or', markersize=3)
  plt.plot(X[y==-1, 0], X[y==-1, 1], 'ob', markersize=3)
  plt.legend(['1','2'],loc="lower left", title="Classes")

  #decision boundary
  symmetry = np.array([X[:,0].min(), X[:,0].max()])
  db = (-W[0] - W[1]*symmetry)/W[2]
  plt.plot(symmetry,db,'--k')

  plt.xlim([-1,0])
  plt.ylim([-1,0])

  plt.title("train result sigmoid")
  plt.xlabel("Symmetry")
  plt.ylabel("Intensity")
  plt.savefig("train_result_sigmoid.png")

  plt.show() 
  ### END YOUR CODE

def visualize_result_multi(X, y, W):
  '''This function is used to plot the softmax model after training. 

  Args:
    X: An array of shape [n_samples, 2].
    y: An array of shape [n_samples,]. Only contains 0,1,2.
    W: An array of shape [n_features, 3].
  
  Returns:
    No return. Save the plot to 'train_result_softmax.*' and include it
    in submission.
  '''
  ### YOUR CODE HERE
  plt.clf()
  plt.plot(X[y==0,0],X[y==0,1],'og',markersize=3)
  plt.plot(X[y==1,0],X[y==1,1],'or',markersize=3)
  plt.plot(X[y==2,0],X[y==2,1],'ob',markersize=3)
  plt.legend(['0','1','2'],loc="lower left", title="Classes")

  symrange = np.linspace(X[:,0].min(), X[:,0].max())
  db1 = np.zeros(symrange.shape)
  db2 = np.zeros(symrange.shape)
  for ix,x1 in enumerate(symrange):
    w0, w1, w2 = (W[0], W[1], W[2])
    db1[ix] = np.max([((w1[0] - w0[0]) + (w1[1] - w0[1])*x1)/(w0[2] - w1[2]), ((w2[0] - w0[0]) + (w2[1] - w0[1])*x1)/(w0[2] - w2[2])])
    db2[ix] = np.min([((w0[0] - w1[0]) + (w0[1] - w1[1])*x1)/(w1[2] - w0[2]), ((w2[0] - w1[0]) + (w2[1] - w1[1])*x1)/(w1[2] - w2[2])])
  plt.plot(symrange,db1,'--k')
  plt.plot(symrange,db2,'--k')
  plt.ylim([-1,0])
  plt.xlim([-1,0])

  plt.title("train result softmax")
  plt.xlabel("Symmetry")
  plt.ylabel("Intensity")
  plt.savefig("train_result_softmax.png")

  plt.show() 

  ### END YOUR CODE

def main():
	# ------------Data Preprocessing------------
	# Read data for training.
    
  raw_data, labels = load_data(os.path.join(data_dir, train_filename))
  raw_train, raw_valid, label_train, label_valid = train_valid_split(raw_data, labels, 2300)

    ##### Preprocess raw data to extract features
  train_X_all = prepare_X(raw_train)
  valid_X_all = prepare_X(raw_valid)
    ##### Preprocess labels for all data to 0,1,2 and return the idx for data from '1' and '2' class.
  train_y_all, train_idx = prepare_y(label_train)
  valid_y_all, val_idx = prepare_y(label_valid)  

    ####### For binary case, only use data from '1' and '2'  
  train_X = train_X_all[train_idx]
  train_y = train_y_all[train_idx]
    ####### Only use the first 1350 data examples for binary training. 
  train_X = train_X[0:1350]
  train_y = train_y[0:1350]
  valid_X = valid_X_all[val_idx]
  valid_y = valid_y_all[val_idx]
    ####### set lables to  1 and -1. Here convert label '2' to '-1' which means we treat data '1' as postitive class. 
  train_y[np.where(train_y==2)] = -1
  valid_y[np.where(valid_y==2)] = -1
  data_shape= train_y.shape[0] 

#    # Visualize training data.
  visualize_features(train_X[:, 1:3], train_y)


   # ------------Logistic Regression Sigmoid Case------------

   ##### Check GD, SGD, BGD
  logisticR_classifier = logistic_regression(learning_rate=0.5, max_iter=100)

  logisticR_classifier.fit_GD(train_X, train_y)
  print(logisticR_classifier.get_params())
  print(logisticR_classifier.score(train_X, train_y))

  logisticR_classifier.fit_BGD(train_X, train_y, data_shape)
  print(logisticR_classifier.get_params())
  print(logisticR_classifier.score(train_X, train_y))

  logisticR_classifier.fit_SGD(train_X, train_y)
  print(logisticR_classifier.get_params())
  print(logisticR_classifier.score(train_X, train_y))

  logisticR_classifier.fit_BGD(train_X, train_y, 1)
  print(logisticR_classifier.get_params())
  print(logisticR_classifier.score(train_X, train_y))

  logisticR_classifier.fit_BGD(train_X, train_y, 10)
  print(logisticR_classifier.get_params())
  print(logisticR_classifier.score(train_X, train_y))


    # Explore different hyper-parameters.
    ### YOUR CODE HERE
    # different learning rate and iteration numbers
  print("----different learning rate and iteration numbers----")
  best_lr = 0.0
  best_it = 0.0
  best_score = 0.0
  lr_list = [0.01, 0.05, 0.1, 0.5, 1]
  it_list = [100, 200, 400, 500, 800, 1000]

  for lr in lr_list: # run loop to record
    for it in it_list:
      x = logistic_regression(learning_rate=lr, max_iter=it)
      x.fit_SGD(train_X, train_y)
      score = x.score(valid_X, valid_y)
      if(score > best_score):
        best_score = score
        best_lr = lr
        best_it = it
        print("score = {}, lr = {}, it = {} \n".format(score, lr, it))
  print("------------------------------")
  print("best_score = {}, best_lr = {}, best_it = {}\n".format(best_score, best_lr, best_it))


  print("----different batch size----")
  best_batch = 1
  batch_list = [4, 16, 64, 128, 256]
  for b in batch_list:
    x = logistic_regression(learning_rate=best_lr, max_iter=best_it)
    x.fit_BGD(train_X, train_y, b)
    score = x.score(valid_X, valid_y)
    if(score > best_score):
      best_score = score
      best_batch = b
      print("score = {}, batch size = {} \n".format(score, b))
  print("------------------------------")
  print("best_batch = {}\n".format(best_batch))
  
  print("\nLogistic Regression---------------")
  best_logisticR = logistic_regression(learning_rate=best_lr, max_iter=best_it)
  best_logisticR.fit_BGD(train_X, train_y, best_batch)
  print("train accuracy: ",best_logisticR.score(train_X,train_y))
  print("validation accuracy: ",best_logisticR.score(valid_X,valid_y))
    ### END YOUR CODE

	# Visualize the your 'best' model after training.
	# visualize_result(train_X[:, 1:3], train_y, best_logisticR.get_params())

    ### YOUR CODE HERE
  visualize_result(train_X[:, 1:3], train_y, best_logisticR.get_params())
    ### END YOUR CODE

    # Use the 'best' model above to do testing. Note that the test data should be loaded and processed in the same way as the training data.
    ### YOUR CODE HERE
  test_data, test_labels = load_data(os.path.join(data_dir, test_filename))

    ##### Preprocess raw data to extract features
  test_X_all = prepare_X(test_data)
    ##### Preprocess labels for all data to 0,1,2 and return the idx for data from '1' and '2' class.
  test_y_all, test_idx = prepare_y(test_labels)

    ####### For binary case, only use data from '1' and '2'  
  test_X = test_X_all[test_idx]
  test_y = test_y_all[test_idx]

    ####### set lables to  1 and -1. Here convert label '2' to '-1' which means we treat data '1' as postitive class. 
  test_y[np.where(test_y == 2)] = -1
  data_shape= test_y.shape[0] 

    ####### get score on test data
  # best_logisticR.fit_BGD(train_valid_X, train_valid_y, best_bs)
  # score = best_logisticR.score(test_X, test_y)
  print("test accuracy: ", best_logisticR.score(test_X, test_y))



    ### END YOUR CODE


    # ------------Logistic Regression Multiple-class case, let k= 3------------
    ###### Use all data from '0' '1' '2' for training
  train_X = train_X_all
  train_y = train_y_all
  valid_X = valid_X_all
  valid_y = valid_y_all

    #########  BGD for multiclass Logistic Regression
  logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=0.5, max_iter=100,  k= 3)
  logisticR_classifier_multiclass.fit_BGD(train_X, train_y, 10)
  print(logisticR_classifier_multiclass.get_params())
  print(logisticR_classifier_multiclass.score(train_X, train_y))

    # Explore different hyper-parameters.
    ### YOUR CODE HERE
    # different learning rate and iteration numbers
  print("----different learning rate and iteration numbers----")
  best_lr = 0.0
  best_it = 0.0
  best_score = 0.0
  lr_list = [0.01, 0.05, 0.1, 0.5, 1]
  it_list = [100, 200, 400, 500, 800, 1000]

  #for lr in lr_list: # run loop to record
  #  for it in it_list:
  #    x = logistic_regression_multiclass(learning_rate=lr, max_iter=it, k=3)
  #    x.fit_BGD(train_X_all, train_y_all, 1)
  #    score = x.score(valid_X_all, valid_y_all)
  #    print("1")
  #    if(score > best_score):
  #      best_score = score
  #      best_lr = lr
  #      best_it = it
  #      print("score = {}, lr = {}, it = {} \n".format(score, lr, it))
  print("------------------------------")
  best_lr = 0.05
  best_it = 100
  x = logistic_regression_multiclass(learning_rate=best_lr, max_iter=best_it, k=3)
  x.fit_BGD(train_X_all, train_y_all, 1)
  best_score = x.score(valid_X_all, valid_y_all)  
  print("best_score = {}, best_lr = {}, best_it = {}\n".format(best_score, best_lr, best_it))


  print("----different batch size----")
  best_batch = 1
  batch_list = [4, 16, 64, 128, 256]
  for b in batch_list:
    x = logistic_regression_multiclass(learning_rate=best_lr, max_iter=best_it, k=3)
    x.fit_BGD(train_X, train_y, b)
    score = x.score(valid_X, valid_y)
    print("2")
    if(score > best_score):
      best_score = score
      best_batch = b
      print("score = {}, batch size = {} \n".format(score, b))
  print("------------------------------")
  print("best_batch = {}\n".format(best_batch))

  print("\nLogistic Regression Multiple-class ---------------")
  best_logistic_multi_R = logistic_regression_multiclass(learning_rate=best_lr, max_iter=best_it, k=3)
  best_logistic_multi_R.fit_BGD(train_X_all, train_y_all, best_batch)
  print("train accuracy: ",best_logistic_multi_R.score(train_X, train_y))
  print("validation accuracy: ",best_logistic_multi_R.score(valid_X, valid_y))
    ### END YOUR CODE

  # Visualize the your 'best' model after training.
  visualize_result_multi(train_X[:, 1:3], train_y, best_logistic_multi_R.get_params())


    # Use the 'best' model above to do testing.
    ### YOUR CODE HERE
  print("test accuracy: ", best_logistic_multi_R.score(test_X_all, test_y_all))
    ### END YOUR CODE


    # ------------Connection between sigmoid and softmax------------
    ############ Now set k=2, only use data from '1' and '2' 

    #####  set labels to 0,1 for softmax classifer
  train_X = train_X_all[train_idx]
  train_y = train_y_all[train_idx]
  train_X = train_X[0:1350]
  train_y = train_y[0:1350]
  valid_X = valid_X_all[val_idx]
  valid_y = valid_y_all[val_idx] 
  train_y[np.where(train_y==2)] = 0
  valid_y[np.where(valid_y==2)] = 0  
    
    ###### First, fit softmax classifer until convergence, and evaluate 
    ##### Hint: we suggest to set the convergence condition as "np.linalg.norm(gradients*1./batch_size) < 0.0005" or max_iter=10000:
    ### YOUR CODE HERE
  maxit = 10000

  print("\n2 Class Softmax---------------")
  softmax_classifer = logistic_regression_multiclass(learning_rate=0.5, max_iter=maxit, k=2)
  softmax_classifer.fit_BGD(train_X, train_y, 128)
  print("train accuracy: ", softmax_classifer.score(train_X, train_y))
  print("validation accuracy: ", softmax_classifer.score(valid_X, valid_y))

  test_X = test_X_all[test_idx]
  test_y = test_y_all[test_idx]
  test_y[np.where(test_y==2)] = 0
  print("test accuracy: ", softmax_classifer.score(test_X, test_y))
    ### END YOUR CODE


  train_X = train_X_all[train_idx]
  train_y = train_y_all[train_idx]
  train_X = train_X[0:1350]
  train_y = train_y[0:1350]
  valid_X = valid_X_all[val_idx]
  valid_y = valid_y_all[val_idx] 
    #####       set lables to -1 and 1 for sigmoid classifer
  train_y[np.where(train_y==2)] = -1
  valid_y[np.where(valid_y==2)] = -1   

    ###### Next, fit sigmoid classifer until convergence, and evaluate
    ##### Hint: we suggest to set the convergence condition as "np.linalg.norm(gradients*1./batch_size) < 0.0005" or max_iter=10000:
    ### YOUR CODE HERE
  print("\n2 Class Sigmoid---------------")    
  sigmoid_classifer = logistic_regression(learning_rate=1, max_iter = maxit)
  sigmoid_classifer.fit_BGD(train_X, train_y, 128)

  print("train accuracy: ", sigmoid_classifer.score(train_X, train_y))
  print("validation accuracy: ", sigmoid_classifer.score(valid_X, valid_y))
  test_X = test_X_all[test_idx]
  test_y = test_y_all[test_idx]
  test_y[np.where(test_y==2)] = -1
  print("test accuracy: ", sigmoid_classifer.score(test_X,test_y))
    ### END YOUR CODE


    ################Compare and report the observations/prediction accuracy


  '''
Explore the training of these two classifiers and monitor the graidents/weights for each step. 
Hint: First, set two learning rates the same, check the graidents/weights for the first batch in the first epoch. What are the relationships between these two models? 
Then, for what leaning rates, we can obtain w_1-w_2= w for all training steps so that these two models are equivalent for each training step. 
  '''
    ### YOUR CODE HERE
  softmax = logistic_regression_multiclass(learning_rate=0.5, max_iter=1, k=2)
  sigmoid = logistic_regression(learning_rate=1, max_iter=1) 

  train_y_multi = train_y.copy()
  train_y_multi[np.where(train_y_multi==-1)] = 0

  sigmoid.fit_BGD(train_X, train_y, 128)
  softmax.fit_BGD(train_X, train_y_multi, 128)
  
  print("sigmoid weight (w): ", sigmoid.get_params())
  print("softmax weight (w1 w2): {}, {}".format(softmax.get_params()[:, 0], softmax.get_params()[:,1]))

    ### END YOUR CODE

    # ------------End------------
    

if __name__ == '__main__':
	main()
    
    
