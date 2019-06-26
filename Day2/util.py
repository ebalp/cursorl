import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
import pandas as pd
from scipy.stats import beta

def plot_2d(X, y=np.array([0])):
    # Plot data
    plt.figure(figsize=(10,10))
    unique_labels = set(y)
    colors = [cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        class_member_mask = (y == k)

        xy = X[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=6, label='{}'.format(k), alpha=0.1)
        
    
    plt.legend(loc='best')
    plt.title('Data')


def plot_regret(solvers, solver_names):
    """
    Plot the regret of contextual bandit solvers.

    Args:
        solvers (list<Solver>): All of them should have been fitted.
        solver_names (list<str>)
    """
    assert len(solvers) == len(solver_names)

    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(111)

    # Sub.fig. 1: Regrets in time.
    for i, s in enumerate(solvers):
        ax1.plot(range(len(s.regrets)), s.regrets, label=solver_names[i])

    ax1.set_xlabel('Time step')
    ax1.set_ylabel('Cumulative regret')
    ax1.legend(loc=9)
    ax1.grid('k', ls='--', alpha=0.3)


def plot_decision_boundary(solver, X, y, h=0.3, scale=1., title='decision boundary'):
    plt.figure(figsize=(10,10))
    unique_labels = set(y)
    colors = {l : cm.Spectral(each) for l , each in zip(unique_labels, np.linspace(0, 1, len(unique_labels)))}

    pan = 1.0
    x_min = X[:, 0].min()
    x_min = x_min - pan if x_min <= 0.0 else x_min + pan
    x_max = X[:, 0].max() + .5
    x_max = x_max - pan if x_max <= 0.0 else x_max + pan
    y_min = X[:, 1].min() - .5
    y_min = y_min - pan if y_min <= 0.0 else y_min + pan
    y_max = X[:, 1].max() + .5
    y_max = y_max - pan if y_max <= 0.0 else y_max + pan
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )

    Z = np.array([solver.action(i) for i in np.c_[xx.ravel(), yy.ravel()]/scale])

    # Put the result into a color plot.
    Z = Z.reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z, cmap=cm.Spectral, alpha=.40, shading= 'gouraud')

    # Add the training points to the plot.
    for k, col in colors.items():
        class_member_mask = (y == k)

        xy = X[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=6, label='{}'.format(k))
    

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.legend(loc='best')
    plt.title(title)
    plt.show()


class ContextualDataset(object):
  """The buffer is able to append new data, and sample random minibatches."""

  def __init__(self, context_dim, num_actions, buffer_s=-1, intercept=False):
    """Creates a ContextualDataset object.

    The data is stored in attributes: contexts and rewards.
    The sequence of taken actions are stored in attribute actions.

    Args:
      context_dim: Dimension of the contexts.
      num_actions: Number of arms for the multi-armed bandit.
      buffer_s: Size of buffer for training. Only last buffer_s will be
        returned as minibatch. If buffer_s = -1, all data will be used.
      intercept: If True, it adds a constant (1.0) dimension to each context X,
        at the end.
    """

    self._context_dim = context_dim
    self._num_actions = num_actions
    self._contexts = None
    self._rewards = None
    self.actions = []
    self.buffer_s = buffer_s
    self.intercept = intercept

  def add(self, context, action, reward):
    """Adds a new triplet (context, action, reward) to the dataset.

    The reward for the actions that weren't played is assumed to be zero.

    Args:
      context: A d-dimensional vector with the context.
      action: Integer between 0 and k-1 representing the chosen arm.
      reward: Real number representing the reward for the (context, action).
    """

    if self.intercept:
      c = np.array(context[:])
      c = np.append(c, 1.0).reshape((1, self.context_dim + 1))
    else:
      c = np.array(context[:]).reshape((1, self.context_dim))

    if self.contexts is None:
      self.contexts = c
    else:
      self.contexts = np.vstack((self.contexts, c))

    r = np.zeros((1, self.num_actions))
    r[0, action] = reward
    if self.rewards is None:
      self.rewards = r
    else:
      self.rewards = np.vstack((self.rewards, r))

    self.actions.append(action)

  def replace_data(self, contexts=None, actions=None, rewards=None):
    if contexts is not None:
      self.contexts = contexts
    if actions is not None:
      self.actions = actions
    if rewards is not None:
      self.rewards = rewards

  def get_batch(self, batch_size):
    """Returns a random minibatch of (contexts, rewards) with batch_size."""
    n, _ = self.contexts.shape
    if self.buffer_s == -1:
      # use all the data
      ind = np.random.choice(range(n), batch_size)
    else:
      # use only buffer (last buffer_s observations)
      ind = np.random.choice(range(max(0, n - self.buffer_s), n), batch_size)
    return self.contexts[ind, :], self.rewards[ind, :]

  def get_data(self, action):
    """Returns all (context, reward) where the action was played."""
    n, _ = self.contexts.shape
    ind = np.array([i for i in range(n) if self.actions[i] == action])
    return self.contexts[ind, :], self.rewards[ind, action]

  @property
  def context_dim(self):
    return self._context_dim

  @property
  def num_actions(self):
    return self._num_actions

  @property
  def contexts(self):
    return self._contexts

  @contexts.setter
  def contexts(self, value):
    self._contexts = value

  @property
  def actions(self):
    return self._actions

  @actions.setter
  def actions(self, value):
    self._actions = value

  @property
  def rewards(self):
    return self._rewards

  @rewards.setter
  def rewards(self, value):
    self._rewards = value
