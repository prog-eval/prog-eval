### Recommender System

The goal of this benchmark is to implement a neural-network based recommender system.  Specifically, the neural network is an autoencoder that takes a list of items that a person has "liked", and predicts a score for all items.  Notice, this implies that the dimension of the input and output layer will be the same.

There are many variations on autoencoders.  The variant we want you to implement is described in the pseudocode below.  Note the nonlinearities (`relu` layers) and regularization techniques (`batchnorm` and `dropout` layers). 

```python

# --
# Initialization

W_emb = rand_normal(num_items, emb_dim, mean=0, std=0.01)

bn1 = batchnorm(dim=emb_dim) # !! Need explanation

scale        = 1 / sqrt(emb_dim)
W_bottleneck = rand_uniform(n_rows=emb_dim, n_cols=hidden_dim, min=-scale, max=scale)
B_bottleneck = zeros(hidden_dim)

bn2 = batchnorm(dim=hidden_dim)

scale       = 1 / sqrt(hidden_dim)
bias_offset = -10
W_output    = rand_uniform(n_rows=hidden_dim, n_cols=num_items, min=-scale, max=scale)
B_output    = zeros(hidden_dim) + bias_offset

# --
# Forward pass of model

def forward(user):
    train_items = get_train_items(user)
    
    emb = zeros(emb_dim)
    for train_item in train_items:
        emb += W_emb[train_item]
    
    h = relu(emb)
    h = bn1(h)
    h = dropout(h, p_zero=0.5)
    
    h = h @ W_bottleneck + B_bottleneck
    
    h = relu(h)
    h = bn2(h)
    h = dropout(h, p_zero=0.5)
    
    h = h @ W_output + B_output
    
    pred = sigmoid(h)
    return pred

```

In case of ambiguity, you can refer to the `pytorch==1.0.0` documentation.

### Train/test splits

The dataset is split into a train set and a test set as follows.  If user U liked items `[I1, I2, ..., I_N]`, we'd 
    - add a record for user U to the train set w/ 80% of those items; and
    - add a record for user U to the validation set w/ the remaining 20% of items
    
In pseudo-code, this looks like
```python
train_items = []
valid_items = []
for user in users:
    items = get_items(user)
    
    user_train_items = sample_without_replacement(items, n=len(items) * 0.80)
    user_valid_items = items.difference(user_train_items)
    
    train_items.append(user_train_items)
    valid_items.append(user_valid_items)
```

You can see the exact procedure in `prep.py`.

### Evaluation

We want the model to assign high scores to items that the user liked.  However, at evaluation time, we will ignore the (user, item) pairs that we've seen in the training set -- we expect the model to (partially) "memorize" those items, so they're not particularly interesting. Thus, before computing our evaluation metrics, we "filter" the training items from the predictions (see `compute_scores` in `helpers.py` for our implementation).  

Thus, given a user and a vector of scores output by the model, we:
    - remove the items seen in the training set for this user
    - rank the remaining items
    - check whether the user liked the item that the model ranks highest

The proportion of top-ranked items that are liked by the user is our metric of interest, `precision-at-1`.

__Wrote quickly -- needs editing for clarity__

#### Targets

Reference code scores
```json
{
    "p_at_01": 0.5148830819428146,
    "p_at_05": 0.40814562523254816, 
    "p_at_10": 0.3438335384527319
}
```
### TODO

- Give better definition of batchnorm
- Can we get rid of ADAM (so that people don't have to maybe implement it)
