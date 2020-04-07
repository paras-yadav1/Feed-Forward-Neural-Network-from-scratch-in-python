import numpy as np
import mnist
import fnn
import check_grad as cg
import matplotlib.pyplot as plt
import time

def preprocess(data):
    with np.errstate(divide='ignore', invalid='ignore'):
        result = (data - data.mean(axis=0)) / data.std(axis=0)
    result[np.isnan(result)] = 0.0
    return result


def create_one_hot_labels(labels, dim=10):
    one_hot_labels = np.zeros((labels.shape[0], dim))
    for i in range(labels.shape[0]):
        one_hot_labels[i][labels[i]] = 1
    return one_hot_labels


def data_preprocessing(data_dir, seed=None):
    # Load mnist data
    mn = mnist.MNIST(data_dir)

    mnist_test_data, mnist_test_labels = mn.load_testing()
    mnist_train_data, mnist_train_labels = mn.load_training()
    raw_test_data = np.array(mnist_test_data)
    raw_train_data = np.array(mnist_train_data)

    # Convert into matrix and one hot vector and preprocess to
    # have mean=0.0 and std=1.0
    mnist_test_data = preprocess(raw_test_data)
    mnist_train_data = preprocess(raw_train_data)
    mnist_test_labels = create_one_hot_labels(np.array(mnist_test_labels))
    mnist_train_labels = create_one_hot_labels(np.array(mnist_train_labels))

    # Split into training and validation set.
    if seed is not None:
        np.random.seed(seed)
    n = mnist_train_data.shape[0]
    indices = np.random.permutation(n)
    n_train = int((55.0/60)*n)
    train_idx, valid_idx = indices[:n_train], indices[n_train:]
    train_data, train_labels = mnist_train_data[train_idx,:], mnist_train_labels[train_idx,:]
    valid_data, valid_labels = mnist_train_data[valid_idx,:], mnist_train_labels[valid_idx,:]

    # Get test set.
    test_data, test_labels = mnist_test_data, mnist_test_labels

    return (train_data, train_labels, valid_data, valid_labels,
            test_data, test_labels, raw_train_data, raw_test_data)


def main():
    print('Loading and preprocessing data\n')
    (train_data, train_labels, valid_data,
     valid_labels, test_data, test_labels,
     raw_train_data, raw_test_data) = data_preprocessing('data/')

    # Initialize model
    print('Initializing neural network\n')
    model = fnn.FNN(784, 10, [128, 32], [fnn.relu, fnn.relu])

    selected = np.random.randint(test_data.shape[0], size=100)
    true_labels = np.argmax(test_labels[selected], axis=1)
    preds_init = model.predict(test_data[selected])

    print('Start training\n')
    n_train = train_data.shape[0]
    n_epochs = 50
    batch_size = 100
    opt = fnn.GradientDescentOptimizer(0.01)

    ########################### YOUR CODE HERE ##########################
    # Write the training loop using the datastructures defined in fnn.py
    # Make sure you study the datastructures and its methods properly
    # before attempting this
    #####################################################################
    n_batches = n_train//batch_size
    indices = np.arange(n_train)
    train_loss=np.zeros(n_epochs)
    valid_loss=np.zeros(n_epochs)
    train_accuracy=np.zeros(n_epochs)
    valid_accuracy=np.zeros(n_epochs)
    start_timer = round(time.time())
    for i in range(n_epochs):
        np.random.shuffle(indices)
        # select batch size samples from the data
        # for j in range(approx total number of data points / batch size):
            # propogate the batch forward
            # use backprop to train your network
            # and use the optimizer to update the model

        for j in range(n_batches):
            train_data_batch = train_data[j*batch_size : (j+1)*batch_size]
            train_labels_batch = train_labels[j*batch_size : (j+1)*batch_size]
            probs, loss= model.forwardprop(train_data_batch, train_labels_batch)
            model.backprop(train_labels_batch)
            opt.update(model)

        # compute the training and validation accuracies and losses and store to use later
        
        (probs_train, loss_train) = model.forwardprop(train_data, train_labels)
        (probs_valid, loss_valid) = model.forwardprop(valid_data, valid_labels)
        
        pred_labels_train = np.argmax(probs_train, axis=1)
        pred_labels_valid = np.argmax(probs_valid, axis=1)
        true_labels_train = np.argmax(train_labels, axis=1)
        true_labels_valid = np.argmax(valid_labels, axis=1)
        
        train_loss[i] = loss_train
        valid_loss[i] = loss_valid
        train_accuracy[i] = np.mean(pred_labels_train == true_labels_train) *100
        valid_accuracy[i] = np.mean(pred_labels_valid == true_labels_valid) *100
      
        print('=' * 20 + ('Epoch %d' % i) + '=' * 20)
        print('Train loss %s accuracy %s\nValid loss %s accuracy %s\n' %
              (train_loss[i], train_accuracy[i], valid_loss[i], valid_accuracy[i]))
        
        
    end_timer = round(time.time())
    time_in_sec = end_timer - start_timer
    
    # Compute test loss and accuracy.
    (probs_test, test_loss) = model.forwardprop(test_data,test_labels)
    
    pred_labels_test = np.argmax(probs_test, axis=1)
    true_labels_test = np.argmax(test_labels, axis=1)
    
    test_accuracy = np.mean(pred_labels_test == true_labels_test) *100
    print('=' * 20 + 'Training finished' + '=' * 20 + '\n')
    print ('Test loss %s accuracy %s\n' %
           (test_loss, test_accuracy))
    print('Training Time = %s min %s sec' %(time_in_sec//60, time_in_sec %60))


    # Graphs 
    fig, axs = plt.subplots(2, 2)

    axs[0,0].plot(range(n_epochs), train_accuracy)
    axs[0,0].set(xlabel='Number of epochs',ylabel='Training accuracy')

    axs[0,1].plot(range(n_epochs), valid_accuracy)
    axs[0,1].set(xlabel='Number of epochs',ylabel='Validation accuracy')

    axs[1,0].plot(range(n_epochs), train_loss)
    axs[1,0].set(xlabel='Number of epochs',ylabel='Training loss')

    axs[1,1].plot(range(n_epochs), valid_loss)
    axs[1,1].set(xlabel='Number of epochs',ylabel='Validation loss')

    fig.tight_layout()
    fig.savefig('plots.png')

    #####################################################################

    preds_trained = model.predict(test_data[selected])

    fig, axes = plt.subplots(10, 10, figsize=(10, 10))
    fig.subplots_adjust(wspace=0)
    for a, image, true_label, pred_init, pred_trained in zip(
            axes.flatten(), raw_test_data[selected],
            true_labels, preds_init, preds_trained):
        a.imshow(image.reshape(28, 28), cmap='gray_r')
        a.text(0, 10, str(true_label), color="black", size=15)
        a.text(0, 26, str(pred_trained), color="blue", size=15)
        a.text(22, 26, str(pred_init), color="red", size=15)

        a.set_xticks(())
        a.set_yticks(())

    plt.show()


if __name__ == '__main__':
    main()
