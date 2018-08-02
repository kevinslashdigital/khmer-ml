
import numpy as np

class Tree(object):
  """
    Tree class is use for building Trees
    and use in DecisionTree Algorithm
  """

  def __init__(self, **kwargs):
    self.feature = None
    self.label = None
    self.n_samples = None
    self.gain = None
    self.left = None
    self.right = None
    self.threshold = None
    self.depth = 0
    self.max_depth = kwargs['max_depth']
    self.kwargs = kwargs

  def build(self, features, target, criterion=None):
    """
      Build a decision tree
    """
    if criterion is None:
      criterion = 'gini'
    # Number of rows of sample
    self.n_samples = features.shape[0]
    # End if all data are same class
    if len(np.unique(target)) == 1:
      self.label = target[0]
      return
    # check for max depth
    if self.depth > self.max_depth:
      return
    best_gain = 0.0
    best_feature = None
    best_threshold = None

    # Classification trees: the most common class, regression tree in the sample: the average in the sample
    if criterion in {'gini', 'entropy', 'error'}:
      # Find the label having high occurences
      self.label = max([(c, len(target[target == c])) for c in np.unique(target)],\
          key=lambda x: x[1])[0]
    else:
        self.label = np.mean(target)
    # Determine node impurity
    impurity_node = self._calc_impurity(criterion, target)
    # Every Attribute, determind the best information gain/gini
    for col_index in range(features.shape[1]):
      # Test split to get the best split
      info_gain, feature_index, threshold = \
      self.test_split(criterion, impurity_node, col_index, features, target)

      # Maximize information gain
      if info_gain > best_gain:
        best_gain = info_gain
        best_feature = feature_index
        best_threshold = threshold
    # No best threshold found,
    # terminate tree building for the current branch
    if best_threshold is None:
      return

    # Start develop left, then right branch
    self.feature = best_feature
    self.gain = best_gain
    self.threshold = best_threshold
    self._divide_tree(features, target, criterion)

  def test_split(self, criterion, impurity_node, index, features, target):
    """
      Split a dataset based on an attribute and an attribute value
    """
    best_gain = 0.0
    best_feature = None
    best_threshold = None
    # Remove redundant features
    feature_level, occurence_freq = np.unique(features[:, index], return_counts=True)
    #elements, occurence_labels = numpy.unique(target_class, return_counts=True)
    # More than one features found
    # Split tree based on "Middle Point"
    if occurence_freq is not None:
      # Pass over the element having frequency zero
      if not feature_level.any():
        return best_gain, best_feature, best_threshold
      if feature_level.shape[0] < 2:
        thresholds = (feature_level[:-1]) / 2.0
      else:
        thresholds = (feature_level[:-1] + feature_level[1:]) / 2.0

      # Explore
      for threshold in thresholds:
        target_l = target[features[:, index] <= threshold]
        impurity_l = self._calc_impurity(criterion, target_l)
        n_l = float(target_l.shape[0]) / self.n_samples

        target_r = target[features[:, index] > threshold]
        impurity_r = self._calc_impurity(criterion, target_r)
        n_r = float(target_r.shape[0]) / self.n_samples

        # (information gain): IG = node - (left + right)
        info_gain = impurity_node - (n_l * impurity_l + n_r * impurity_r)

        # Maximize information gain
        if info_gain > best_gain:
          best_gain = info_gain
          best_feature = index
          best_threshold = threshold
    return best_gain, best_feature, best_threshold

  def _divide_tree(self, features, target, criterion):

    features_l = features[features[:, self.feature] <= self.threshold]
    target_l = target[features[:, self.feature] <= self.threshold]
    #self.left = Tree(self.max_depth)
    self.left = Tree(**self.kwargs)
    self.left.depth = self.depth + 1
    self.left.build(features_l, target_l, criterion)

    features_r = features[features[:, self.feature] > self.threshold]
    target_r = target[features[:, self.feature] > self.threshold]
    self.right = Tree(**self.kwargs)
    self.right.depth = self.depth + 1
    self.right.build(features_r, target_r, criterion)

  def _calc_impurity(self, criterion, target):
    # Number of classes, and
    # Number of sample classes
    unique_target = np.unique(target)
    n_rows = target.shape[0]

    if criterion == 'gini':
      return self._gini(target, unique_target, n_rows)
    elif criterion == 'entropy':
      return self._entropy(target, unique_target, n_rows)
    elif criterion == 'error':
      return self._error(target, unique_target, n_rows)
    elif criterion == 'mse':
      return self._mse(target)
    else:
      return self._gini(target, unique_target, n_rows)

  # Gini impurity
  def _gini(self, target, n_classes, n_samples):
    """
      Calculate the Gini index
    """
    gini_index = 1.0
    gini_index -= sum([(float(len(target[target == label])) / float(n_samples)) ** 2.0 for label in n_classes])
    return gini_index

  # Entropy
  def _entropy(self, target, n_classes, n_samples):
    """
      Calculate the entropy of a dataset.
      The only parameter of this function is the target_col parameter which specifies the target column
    """
    entropy = 0.0
    # elements: each unique label in the label array
    # counts: the number of times each unique label appears in label array
    for label in n_classes:
      prob = float(len(target[target == label])) / n_samples
      if prob > 0.0:
        entropy -= prob * np.log2(prob)
    return entropy

  def _error(self, target, n_classes, n_samples):
    """
      Classification error
    """
    return 1.0 - max([len(target[target == label]) / n_samples for label in n_classes])

  def _mse(self, target):
    """
      Mean square error
    """
    y_hat = np.mean(target)
    return np.mean((target - y_hat) ** 2.0)

  def prune(self, method, limit_depth, min_criterion, n_samples):
    """
      Prune decision tree
    """

    if self.feature is None:
      return

    self.left.prune(method, limit_depth, min_criterion, n_samples)
    self.right.prune(method, limit_depth, min_criterion, n_samples)

    pruning = False

    # Pruning decision, Leaf
    if method == 'impurity' and self.left.feature is None and self.right.feature is None:
      if (self.gain * float(self.n_samples) / n_samples) < min_criterion:
        pruning = True
    elif method == 'depth' and self.depth >= limit_depth:
      pruning = True

    # Pruning suppresses over learning
    if pruning is True:
      self.left = None
      self.right = None
      self.feature = None


  def _predict(self, X_test):
    """
      Prediction of a new/unseen query instance.
    """
    # print('self.feature',self.feature,X_test)
    # Node
    if self.feature is not None:
      if X_test[self.feature] <= self.threshold:
        return self.left._predict(X_test)
      else:
        return self.right._predict(X_test)
    else: # Leaf
      # print(' self.label', self.label)
      return self.label

  def show_tree(self, depth, cond):
    """
      Display the structure of built tree from learning process
    """
    base = '    ' * depth + cond
    if self.feature != None: # Node
      print(base + 'if X[' + str(self.feature) + '] <= ' + str(self.threshold))
      self.left.show_tree(depth+1, 'then ')
      self.right.show_tree(depth+1, 'else ')
    else: # Leaf
      print(base + '{value: ' + str(self.label) + ', samples: ' + str(self.n_samples) + '}')
