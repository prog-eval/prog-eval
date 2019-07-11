# graphsage

__Task:__ Implement and train the [GraphSAGE](http://snap.stanford.edu/graphsage/) neural network algorithm, which is used to do deep learning on graph-structured data.

Consider a "citation network", where nodes represent academic papers and edges indicate that one paper cites another.  Suppose we want to train a machine learning model to classify the domain area of each paper.  A straightforward approach would be to train a model based on word counts in the paper -- if a paper says "microchip" many times, it's likely about electrical engineering, but if it says "tumor" then it's more likely about biology.  However, this approach is ignoring all of the _network_ information: electrical engineering papers are more likely to be cited by electrical engineering papers, biology papers more likely to be cited by biology papers.  GraphSAGE is a neural network algorithm that can take advantage of both the "content" of a paper (eg. word counts) as well as the "context" (eg. neighbors in the citation network).

Roughly, when we want to make a prediction for a particular node `u` in the network, GraphSAGE aggregates information from a (sampled)neighborhood of `u`.  That is, it randomly picks a subset of nodes that are one-hop neighbors of `u`, then randomly picks a subset of one-hop neighbors for each of those nodes, and runs all of their features through a neural network to get a prediction for `u`.

Please read the GraphSAGE [paper](https://arxiv.org/pdf/1706.02216.pdf) for a more detailed introduction to the algorithm.  

## Algorithm Overview

The paper describes a number of variants of the algorithm.  We want you to implement the specific variant shown below, which (roughly) corresponds to a `depth=2` GraphSAGE architecture with `relu` nonlinearities and `mean` aggregators.

```python

# --
# Trainable parameters

scale     = sqrt(1 / 2 * feature_dim)
W_encoder = rand_uniform(nrows=2 * feature_dim, ncols=hidden_dim, min=-scale, max=scale)

scale     = sqrt(1 / 2 * hidden_dim)
W_hidden1 = rand_uniform(nrows=2 * hidden_dim, ncols=hidden_dim, min=-scale, max=scale)
b_hidden1 = zeros(hidden_dim)

scale     = sqrt(1 / hidden_dim)
W_hidden0 = rand_uniform(nrows=hidden_dim, ncols=hidden_dim, min=-scale, max=scale)
b_hidden0 = zeros(hidden_dim)

scale = sqrt(1 / hidden_dim)
W_out = rand_uniform(nrows=hidden_dim, ncols=1, min=-scale, max=scale)
b_out = zeros(1)
# !! Note: ncols=1 because this is a regression model

# --
# Pseudocode

def embed_node(node):
    
    # Get mean representation of 1-hop neighbors
    avg_1hop = zeros(hidden_dim) 
    for neib_1hop in node.sample_neighbors(num_neighbors):
        
        # Get encoding of 1 hop neighbor
        enc_1hop = relu(features[neib_1hop] @ W_encoder)
        
        # Get mean encoding of 2 hop neighbors
        avg_2hop = zeros(hidden_dim)
        for neib_2hop in neib_1hop.sample_neighbors(num_neighbors):
            avg_2hop += relu(features[neib_2hop] @ W_encoder)
        
        avg_2hop /= num_neighbors
        
        # Combine 1- and 2-hop encodings
        hidden_state1 = concatenate(enc_1hop, avg_2hop)
        hidden_state1 = hidden_state1 @ W_hidden1 + b_hidden1
        hidden_state1 = relu(hidden_state1)
        
        avg_1hop += hidden_state1
    
    avg_1hop /= num_neighbors
    
    # Another nonlinear layer
    hidden_state0 = avg_1hop @ W_hidden0 + b_hidden0
    hidden_state0 = relu(hidden_state0)
    
    prediction = hidden_state0 @ W_out + b_out
    return prediction
```

Some relevant parameters are given below (these are also visible in `parse_args` in the code skeleton)
  - Batch size: 256
  - Number of epochs (passes over training data): 3
  - Number of neighbors: 12
  - Optimizer: SGD optimizer (as implemented in `pytorch==1.0.0`)
    - Learning rate: Reduce learning rate linearly from 0.01 to 0 over 3 epochs (updating each minibatch, rather than each epoch)
      - This kind of "learning rate annealing" is a trick that helps models converge more quickly
    - Nesterov momentum: 0.9
    - If you need to implement your own optimizer, you can look at [pytorch](https://pytorch.org/docs/stable/_modules/torch/optim/sgd.html) as a reference.  Simplified versions of their optimizers can be seen [in this gist](https://gist.github.com/bkj/77bf8eabb52b1dfac41c69085e07fd3d)
  - Loss function: mean absolute error `mean(abs(y - y_hat))`
    - __This is a regression problem__
  - Hidden dimension: 64
  - Initializers: 
    - Embedding layer: Normal(mean=0, std=1)
    - Linear layers: Uniform(min=-1 / sqrt(input_dim), max=1 / sqrt(input_dim))


__Note:__ The _exact_ details of how the SGD optimizer implemented probably shouldn't matter -- Tensorflow implementations may be slightly different than Pytorch implementations may be different than custom implementations, but the quality of the model should be the same.  If you _strongly_ suspect that you're having trouble that's tied to either of these parts of the model, please reach out for support.

## Evaluation

We want to 

  - minimize the mean absolute error between predictions and ground truth, and
  - maximize the Pearson correlation between predictions and ground truth

on the validation dataset.

__To be considered "correct", your implementation must achieve `MAE < 3.9` and `pearson_correlation > 0.72` after three epochs of training.__

(__Hint:__ If correctly implemented, you should certainly be seeing `MAE < 4` and `pearson_correlation > 0.6` after the first epoch.)

## Notes

Implemented in pytorch, our reference implementation
 - takes 2-3 minutes on a CPU w/ 8 threads (Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz)
 - takes about 30 seconds on a GPU (P100)

## References

- [1] Inductive Representation Learning on Large Graphs: https://arxiv.org/pdf/1706.02216.pdf
- [2] The task we're testing comes from [this short paper](http://perozzi.net/publications/15_www_age.pdf).
