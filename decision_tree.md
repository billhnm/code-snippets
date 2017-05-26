# Code to fix target
The target column (label column) of the dataset that we are interested in is called bad_loans. In this column 1 means a risky (bad) loan 0 means a safe loan.

In order to make this more intuitive and consistent with the lectures, we reassign the target to be:

   * +1 as a safe loan,
   * -1 as a risky (bad) loan.

We put this in a new column called safe_loans.
* safe_loans =  1 => safe
* safe_loans = -1 => risky

  loans['safe_loans'] = loans['bad_loans'].apply(lambda x : +1 if x==0 else -1)
  loans = loans.remove_column('bad_loans')

# Select features and add target
features = ['grade',                     # grade of the loan
            'sub_grade',                 # sub-grade of the loan
            'short_emp',                 # one year or less of employment
            'emp_length_num',            # number of years of employment
            'home_ownership',            # home_ownership status: own, mortgage or rent
            'dti',                       # debt to income ratio
            'purpose',                   # the purpose of the loan
            'term',                      # the term of the loan
            'last_delinq_none',          # has borrower had a delinquincy
            'last_major_derog_none',     # has borrower had 90 day or worse rating
            'revol_util',                # percent of available credit being used
            'total_rec_late_fee',        # total late fees received to day
           ]

target = 'safe_loans'                   # prediction target (y) (+1 means safe, -1 is risky)

# Extract the feature columns and target column
loans = loans[features + [target]]

# Code to sub-sample majority class to even out results
# Since there are fewer risky loans than safe loans, find the ratio of the sizes
# and use that percentage to undersample the safe loans.
percentage = len(risky_loans_raw)/float(len(safe_loans_raw))

risky_loans = risky_loans_raw
safe_loans = safe_loans_raw.sample(percentage, seed=1)

# Append the risky_loans with the downsampled version of safe_loans
loans_data = risky_loans.append(safe_loans)

# Graphlab Create version:
decision_tree_model = graphlab.decision_tree_classifier.create(train_data, validation_set=None,
                                target = target, features = features)

# sklearn version
# see: http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html 

# show the tree
model.show(view="Tree") 

# Save predictions to an SArray. 
predictions = decision_tree_model.predict(sample_validation_data)
predictions
# Evaluate the model and save the results into a dictionary
# results = model.evaluate(test_data)

# get probability predictions
# pred_prob_one = model.predict(test_data, output_type = 'probability') 
prob_predictions = decision_tree_model.predict(sample_validation_data, output_type = 'probability')
prob_predictions

# get accuracy predictions
accuracy_predictions = model.evaluate(train_data)['accuracy']

# check for false positives and false negatives
false_positives = []
for i in range(len(predictions)):
    if (predictions[i] == 1) & (actual_loans[i] == -1):
        false_positives.append(1)
    else:
        false_positives.append(0)
        
sum(false_positives)

false_negatives = []
for i in range(len(predictions)):
    if (predictions[i] == -1) & (actual_loans[i] == 1):
        false_negatives.append(1)
    else:
        false_negatives.append(0)
        
sum(false_negatives)

# 
# This section for decision tree from scratch
#

# count number of mistakes at a node
def intermediate_node_num_mistakes(labels_in_node):
    # Corner case: If labels_in_node is empty, return 0
    if len(labels_in_node) == 0:
        return 0
    
    # Count the number of 1's (safe loans)
    ## YOUR CODE HERE
    number_safe = sum(labels_in_node == 1)
    
    # Count the number of -1's (risky loans)
    ## YOUR CODE HERE
    number_risky = sum(labels_in_node == -1)   
        
    # Return the number of mistakes that the majority classifier makes.
    ## YOUR CODE HERE
    if number_safe > number_risky:
        mistakes = number_risky
        # print "risky"
    else:
        mistakes = number_safe
        # print "safe"
    return mistakes
    
# pick best feature to split on
def best_splitting_feature(data, features, target):
    
    best_feature = None # Keep track of the best feature 
    best_error = 10     # Keep track of the best error so far 
    # Note: Since error is always <= 1, we should intialize it with something larger than 1.

    # Convert to float to make sure error gets computed correctly.
    num_data_points = float(len(data))  
    
    # Loop through each feature to consider splitting on that feature
    for feature in features:
        
        # The left split will have all data points where the feature value is 0
        left_split = data[data[feature] == 0]
        
        # The right split will have all data points where the feature value is 1
        ## YOUR CODE HERE
        right_split = data[data[feature] == 1]
            
        # Calculate the number of misclassified examples in the left split.
        # Remember that we implemented a function for this! (It was called intermediate_node_num_mistakes)
        # YOUR CODE HERE
        left_mistakes = intermediate_node_num_mistakes(left_split[target])          

        # Calculate the number of misclassified examples in the right split.
        ## YOUR CODE HERE
        right_mistakes = intermediate_node_num_mistakes(right_split[target])
            
        # Compute the classification error of this split.
        # Error = (# of mistakes (left) + # of mistakes (right)) / (# of data points)
        ## YOUR CODE HERE
        error = (left_mistakes + right_mistakes) / num_data_points

        # If this is the best error we have found so far, store the feature as best_feature and the error as best_error
        ## YOUR CODE HERE
        if error < best_error:
            best_feature = feature
            best_error = error
            
    return best_feature # Return the best feature we found
    
# create a new leaf
def create_leaf(target_values):
    
    # Create a leaf node
    leaf = {'splitting_feature' : None,
            'left' : None,
            'right' : None,
            'is_leaf': True    }   ## YOUR CODE HERE
    
    # Count the number of data points that are +1 and -1 in this node.
    num_ones = len(target_values[target_values == +1])
    num_minus_ones = len(target_values[target_values == -1])
    
    # For the leaf node, set the prediction to be the majority class.
    # Store the predicted class (1 or -1) in leaf['prediction']
    if num_ones > num_minus_ones:
        leaf['prediction'] = 1        ## YOUR CODE HERE
    else:
        leaf['prediction'] = -1        ## YOUR CODE HERE
        
    # Return the leaf node        
    return leaf 

# Create decision tree with basic stopping conditions
def decision_tree_create(data, features, target, current_depth = 0, max_depth = 10):
    remaining_features = features[:] # Make a copy of the features.
    
    target_values = data[target]
    print "--------------------------------------------------------------------"
    print "Subtree, depth = %s (%s data points)." % (current_depth, len(target_values))
    

    # Stopping condition 1
    # (Check if there are mistakes at current node.
    # Recall you wrote a function intermediate_node_num_mistakes to compute this.)
    mistakes = intermediate_node_num_mistakes(data[target])
    if mistakes == 0:  ## YOUR CODE HERE
        print "Stopping condition 1 reached."     
        # If not mistakes at current node, make current node a leaf node
        return create_leaf(target_values)
    
    # Stopping condition 2 (check if there are remaining features to consider splitting on)
    if remaining_features == False:   ## YOUR CODE HERE
        print "Stopping condition 2 reached."    
        # If there are no remaining features to consider, make current node a leaf node
        return create_leaf(target_values)    
    
    # Additional stopping condition (limit tree depth)
    if current_depth >= max_depth:  ## YOUR CODE HERE
        print "Reached maximum depth. Stopping for now."
        # If the max tree depth has been reached, make current node a leaf node
        return create_leaf(target_values)

    # Find the best splitting feature (recall the function best_splitting_feature implemented above)
    ## YOUR CODE HERE
    
    splitting_feature = best_splitting_feature(data, remaining_features, target)

    
    # Split on the best feature that we found. 
    left_split = data[data[splitting_feature] == 0]
    right_split = data[data[splitting_feature] == 1]      ## YOUR CODE HERE
    remaining_features.remove(splitting_feature)
    print "Split on feature %s. (%s, %s)" % (\
                      splitting_feature, len(left_split), len(right_split))
    
    # Create a leaf node if the split is "perfect"
    if len(left_split) == len(data):
        print "Creating leaf node."
        return create_leaf(left_split[target])
    if len(right_split) == len(data):
        print "Creating leaf node."
        return create_leaf(right_split[target])

        
    # Repeat (recurse) on left and right subtrees
    left_tree = decision_tree_create(left_split, remaining_features, target, current_depth + 1, max_depth)        
    ## YOUR CODE HERE
    right_tree = decision_tree_create(right_split, remaining_features, target, current_depth + 1, max_depth)

    return {'is_leaf'          : False, 
            'prediction'       : None,
            'splitting_feature': splitting_feature,
            'left'             : left_tree, 
            'right'            : right_tree}

# Function to count nodes in the tree
def count_nodes(tree):
    if tree['is_leaf']:
        return 1
    return 1 + count_nodes(tree['left']) + count_nodes(tree['right'])

# function to return classification from tree
def classify(tree, x, annotate = False):   
    # if the node is a leaf node.
    if tree['is_leaf']:
        if annotate: 
            print "At leaf, predicting %s" % tree['prediction']
        return tree['prediction'] 
    else:
        # split on feature.
        split_feature_value = x[tree['splitting_feature']]
        if annotate: 
            print "Split on %s = %s" % (tree['splitting_feature'], split_feature_value)
        if split_feature_value == 0:
            return classify(tree['left'], x, annotate)
        else:
            return classify(tree['right'], x, annotate) 
            # return tree['prediction'] ### YOUR CODE HERE

# function to evaluate classification error
def evaluate_classification_error(tree, data, target):
    # Apply the classify(tree, x) to each row in your data
    prediction = data.apply(lambda x: classify(tree, x))
    
    # Once you've made the predictions, calculate the classification error and return it
    ## YOUR CODE HERE
    num_data_points = float(len(data)) 
    mistakes = sum(prediction != data[target])
    classification_error = mistakes / num_data_points
    return classification_error

# function to print out a stump of a decision tree
def print_stump(tree, name = 'root'):
    split_name = tree['splitting_feature'] # split_name is something like 'term. 36 months'
    if split_name is None:
        print "(leaf, label: %s)" % tree['prediction']
        return None
    split_feature, split_value = split_name.split('.')
    print '                       %s' % name
    print '         |---------------|----------------|'
    print '         |                                |'
    print '         |                                |'
    print '         |                                |'
    print '  [{0} == 0]               [{0} == 1]    '.format(split_name)
    print '         |                                |'
    print '         |                                |'
    print '         |                                |'
    print '    (%s)                         (%s)' \
        % (('leaf, label: ' + str(tree['left']['prediction']) if tree['left']['is_leaf'] else 'subtree'),
           ('leaf, label: ' + str(tree['right']['prediction']) if tree['right']['is_leaf'] else 'subtree'))

# note: can use to access any node
# Root
print_stump(my_decision_tree)

# 1st level
print_stump(my_decision_tree['left'], my_decision_tree['splitting_feature'])

# 2nd level
print_stump(my_decision_tree['left']['left'], my_decision_tree['left']['splitting_feature'])


