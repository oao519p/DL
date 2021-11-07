from DataReader import prepare_data
from model import Model

data_dir = "../data/"
train_filename = "training.npz"
test_filename = "test.npz"

def main():
    # ------------Data Preprocessing------------
    train_X, train_y, valid_X, valid_y, train_valid_X, train_valid_y, test_X, test_y = prepare_data(data_dir, train_filename, test_filename)

    # ------------Kernel Logistic Regression Case------------
    ### YOUR CODE HERE
    # Run your kernel logistic regression model here
    learning_rate = [0.01, 0.001]
    max_epoch = [10,30,50]
    batch_size = train_valid_X.shape[0]
    sigma = [0.1, 1,5,10,15,18,20,40]

    #ans = [] 
	
    #for lr in learning_rate:  # test hyper-parameters
    #    for me in max_epoch:
    #        for sig in sigma: 
    #            model = Model('Kernel_LR', train_X.shape[0], sig)
    #            model.train(train_X, train_y, valid_X, valid_y, me, lr, batch_size)
    #            score = model.score(test_X, test_y)
    #            print("score = {} in test set.\n".format(score))

    #            ans.append((lr, me, sig, score))	

    #for a in ans:
    #    print(a)

    learning_rate = 0.01
    max_epoch = 50
    sigma = 1

    model = Model('Kernel_LR', train_valid_X.shape[0], sigma)
    model.train(train_valid_X, train_valid_y, None, None, max_epoch, learning_rate, batch_size)
    score = model.score(test_X, test_y)
    print("------------Kernel Logistic Regression Case------------")
    print("score = {} in test set.\n".format(score))
    ### END YOUR CODE

    # ------------RBF Network Case------------
    ### YOUR CODE HERE
    # Run your radial basis function network model here
    #hidden_dim = [1,2,4,8,16,32]
    learning_rate = 0.01
    max_epoch = 50
    batch_size = 64
    #sigma = [0.1,1,5,10,15,18,20,40]

    #ans = [] 
	
    #for hd in hidden_dim:  # test hyper-parameters
    #  for sig in sigma: 
    #    model = Model('RBF', hd, sig)
    #    model.train(train_X, train_y, valid_X, valid_y, max_epoch, learning_rate, batch_size)
             
    #    model = Model('RBF', hd, sig)
    #    model.train(train_valid_X, train_valid_y, None, None, max_epoch, learning_rate, batch_size)
    #    score = model.score(test_X, test_y)
    #    print("score = {} in test set.\n".format(score))

    #    ans.append((hd, sig, score))	

    #for a in ans:
    #  print(a) 


    hidden_dim = 4
    sigma = 1
    model = Model('RBF', hidden_dim, sigma)
    model.train(train_X, train_y, valid_X, valid_y, max_epoch, learning_rate, batch_size)

    model = Model('RBF', hidden_dim, sigma)
    model.train(train_valid_X, train_valid_y, None, None, max_epoch, learning_rate, batch_size)
    score = model.score(test_X, test_y)
    print("------------RBF Network Case------------")
    print("score = {} in test set.\n".format(score))
    ### END YOUR CODE

    # ------------Feed-Forward Network Case------------
    ### YOUR CODE HERE
    # Run your feed-forward network model here
    #hidden_dim = [1,2,4,8,16,32]
    learning_rate = 0.01
    max_epoch = 50
    batch_size = 64

    #ans = [] 
	
    #for hd in hidden_dim:  # test hyper-parameters
    #  model = Model('FFN', hd)
    #  model.train(train_X, train_y, valid_X, valid_y, max_epoch, learning_rate, batch_size)
    #  model = Model('FFN', hd)
    #  model.train(train_valid_X, train_valid_y, None, None, max_epoch, learning_rate, batch_size)
    #  score = model.score(test_X, test_y)
    #  print("score = {} in test set.\n".format(score))

    #  ans.append((hd, score))	

    #for a in ans:
    #  print(a)


    hidden_dim = 4
    model = Model('FFN', hidden_dim)
    model.train(train_X, train_y, valid_X, valid_y, max_epoch, learning_rate, batch_size)

    model = Model('FFN', hidden_dim)
    model.train(train_valid_X, train_valid_y, None, None, max_epoch, learning_rate, batch_size)
    score = model.score(test_X, test_y)
    print("------------Feed-Forward Network Case------------")
    print("score = {} in test set\n".format(score))
    ### END YOUR CODE
    
if __name__ == '__main__':
    main()