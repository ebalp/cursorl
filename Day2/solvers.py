import numpy as np
import time
import tensorflow as tf

from scipy.stats import invgamma

from Day2.bandits import ContextualBandit
from Day2.util import ContextualDataset


class Solver(object):
    def __init__(self, bandit):
        """
        bandit (Bandit): the target bandit to solve.
        """
        np.random.seed(int(time.time()))

        self.bandit = bandit

        self.actions = []  # A list of actions
        self.rewards = []
        self.regret = 0.  # Cumulative regret.
        self.regrets = [0.]  # History of cumulative regret.

    def update_regret(self, i):
        # i (int): index of the selected machine.
        self.regret += self.bandit.best_reward - self.bandit.rewards[i]
        self.regrets.append(self.regret)



class Random(Solver):
    def __init__(self, bandit):

        super(Random, self).__init__(bandit)
    
    def action(self, context):
        return np.random.randint(0, self.bandit.n)

    def update(self, context, action, reward):
        self.actions.append(action)
        self.rewards.append(reward)

        self.regret += self.bandit.best_reward(context) - reward
        self.regrets.append(self.regret)
    


class EpsilonGreedyNeural(Solver):
    def __init__(self, bandit, eps, architecture, hparams=dict()):
        """
        eps (float): the probability to explore at each time step.
        architecture (keras Secuential): Compliled Neural Network model, last layer shape should match number of actions. 
        """
        super(EpsilonGreedyNeural, self).__init__(bandit)

        assert architecture.layers[-1].output_shape[1] == bandit.n
    
        self.step = 0
        self.bandit = bandit
        self.eps = eps
        self.model = architecture

        self.batch_size = hparams.get('batch_size', 64)
        self.training_freq = hparams.get('training_freq', 25)
        self.training_epochs = hparams.get('training_epochs', 20)
        self.init_pulls = hparams.get('initial_pulls', 2)
        self.context_dim = bandit.context_dim
        self.num_actions = bandit.n
        self.data_h = ContextualDataset(bandit.context_dim, bandit.n, hparams.get('buffer_s', -1))

    def action(self, context):
        """Selects greedy action."""

        if self.step < self.num_actions * self.init_pulls:
            # round robin until each action has been taken "initial_pulls" times
            return self.step % self.num_actions

        if np.random.random() < self.eps:
            # Let's do random exploration!
            return np.random.randint(0, self.bandit.n)
        
        c = context.reshape((1, self.context_dim))
        o = self.model.predict(c)
        return np.argmax(o)

    def update(self, context, action, reward):
        """Updates data buffer, and re-trains the network every training_freq steps."""
        self.step += 1
        self.data_h.add(context, action, reward)

        self.actions.append(action)
        self.rewards.append(reward)

        self.regret += self.bandit.best_reward(context) - reward
        self.regrets.append(self.regret)

        if self.step % self.training_freq == 0:
            for _ in range(self.training_epochs):
                x, y = self.data_h.get_batch(self.batch_size)
                self.model.fit(x, y, epochs=1, verbose=0)



class BayesianLinear(Solver):
    def __init__(self, bandit, hparams=dict()):
        """

        """
        super(BayesianLinear, self).__init__(bandit)
    
        self.step = 0
        self.bandit = bandit

        self.init_pulls = hparams.get('initial_pulls', 2)
        self.context_dim = bandit.context_dim
        self.num_actions = bandit.n
        self.data_h = ContextualDataset(bandit.context_dim, bandit.n, hparams.get('buffer_s', -1), intercept=True)

        # Gaussian prior for each beta_i
        self.lambda_prior = hparams.get('lambda_prior', 0.25)

        self.mu = [
            np.zeros(self.context_dim + 1)
            for _ in range(self.num_actions)
        ]

        self.cov = [(1.0 / self.lambda_prior) * np.eye(self.context_dim + 1)
                    for _ in range(self.num_actions)]

        self.precision = [
            self.lambda_prior * np.eye(self.context_dim + 1)
            for _ in range(self.num_actions)
        ]

        # Inverse Gamma prior for each sigma2_i
        self.a0 = hparams.get('a0', 6)
        self.b0 = hparams.get('b0', 6)

        self.a = [self.a0 for _ in range(self.num_actions)]
        self.b = [self.b0 for _ in range(self.num_actions)]


    def action(self, context):
        """Samples beta's from posterior, and chooses best action accordingly.

        Args:
        context: Context for which the action need to be chosen.

        Returns:
        action: Selected action for the context.
        """

        # Round robin until each action has been selected "initial_pulls" times
        if self.step < self.num_actions * self.init_pulls:
            return self.step % self.num_actions

        # Sample sigma2, and beta conditional on sigma2
        sigma2_s = [
            self.b[i] * invgamma.rvs(self.a[i])
            for i in range(self.num_actions)
        ]

        try:
            beta_s = [
                np.random.multivariate_normal(self.mu[i], sigma2_s[i] * self.cov[i])
                for i in range(self.num_actions)
            ]
        except np.linalg.LinAlgError as e:
            # Sampling could fail if covariance is not positive definite
            print('Exception when sampling')
            print('Details: {}'.format(e.args))
            d = self.context_dim + 1
            beta_s = [
                np.random.multivariate_normal(np.zeros((d)), np.eye(d))
                for i in range(self.num_actions)
            ]

        # Compute sampled expected values, intercept is last component of beta
        vals = [
            np.dot(beta_s[i][:-1], context.T) + beta_s[i][-1]
            for i in range(self.num_actions)
        ]

        return np.argmax(vals) #Index of maximum value

    def update(self, context, action, reward):
        """Updates action posterior using the linear Bayesian regression formula.

        Args:
        context: Last observed context.
        action: Last observed action.
        reward: Last observed reward.
        """

        self.step += 1
        self.data_h.add(context, action, reward)

        self.actions.append(action)
        self.rewards.append(reward)

        self.regret += self.bandit.best_reward(context) - reward
        self.regrets.append(self.regret)

        # Update posterior of action with formulas: \beta | x,y ~ N(mu_q, cov_q)
        x, y = self.data_h.get_data(action)

        # The algorithm could be improved with sequential update formulas (cheaper)
        s = np.dot(x.T, x)

        # Some terms are removed as we assume prior mu_0 = 0.
        precision_a = s + self.lambda_prior * np.eye(self.context_dim + 1)
        cov_a = np.linalg.inv(precision_a)
        mu_a = np.dot(cov_a, np.dot(x.T, y))

        # Inverse Gamma posterior update
        a_post = self.a0 + x.shape[0] / 2.0
        b_upd = 0.5 * (np.dot(y.T, y) - np.dot(mu_a.T, np.dot(precision_a, mu_a)))
        b_post = self.b0 + b_upd

        # Store new posterior distributions
        self.mu[action] = mu_a
        self.cov[action] = cov_a
        self.precision[action] = precision_a
        self.a[action] = a_post
        self.b[action] = b_post



class NeuralLinear(Solver):
    def __init__(self, bandit, architecture, hparams=dict()):
        """
        architecture (keras Secuential): Compliled Neural Network model, last layer shape should match number of actions. 
        """
        super(NeuralLinear, self).__init__(bandit)

        assert architecture.layers[-1].output_shape[1] == bandit.n
    
        self.step = 0
        self.bandit = bandit
        self.model = architecture
        self.model_hl = tf.keras.Model(inputs=architecture.input, outputs=architecture.get_layer(index=-2).output)
        self.latent_dim = architecture.layers[-2].output_shape[1]

        self.batch_size = hparams.get('batch_size', 64)
        self.training_epochs = hparams.get('training_epochs', 20)
        self.init_pulls = hparams.get('initial_pulls', 2)

        # Regression and NN Update Frequency
        self.update_freq_lr = hparams.get('training_freq_lr', 1)
        self.update_freq_nn = hparams.get('training_freq_network', 25)

        self.context_dim = bandit.context_dim
        self.num_actions = bandit.n
        self.data_h = ContextualDataset(bandit.context_dim, bandit.n, hparams.get('buffer_s', -1), intercept=False)
        self.latent_h = ContextualDataset(self.latent_dim, bandit.n, hparams.get('buffer_s', -1), intercept=False)

        # Gaussian prior for each beta_i
        self.lambda_prior = hparams.get('lambda_prior', 0.25)

        self.mu = [
            np.zeros(self.context_dim + 1)
            for _ in range(self.num_actions)
        ]

        self.cov = [(1.0 / self.lambda_prior) * np.eye(self.context_dim + 1)
                    for _ in range(self.num_actions)]

        self.precision = [
            self.lambda_prior * np.eye(self.context_dim + 1)
            for _ in range(self.num_actions)
        ]

        # Inverse Gamma prior for each sigma2_i
        self.a0 = hparams.get('a0', 6)
        self.b0 = hparams.get('b0', 6)

        self.a = [self.a0 for _ in range(self.num_actions)]
        self.b = [self.b0 for _ in range(self.num_actions)]


    def action(self, context):
        """Samples beta's from posterior, and chooses best action accordingly."""

        # Round robin until each action has been selected "initial_pulls" times
        if self.step < self.num_actions * self.init_pulls:
            return self.step % self.num_actions

        # Sample sigma2, and beta conditional on sigma2
        sigma2_s = [
            self.b[i] * invgamma.rvs(self.a[i])
            for i in range(self.num_actions)
        ]

        try:
            beta_s = [
                np.random.multivariate_normal(self.mu[i], sigma2_s[i] * self.cov[i])
                for i in range(self.num_actions)
            ]
        except np.linalg.LinAlgError as e:
            # Sampling could fail if covariance is not positive definite
            print('Exception when sampling')
            print('Details: {}.'.format(e.args))
            d = self.latent_dim
            beta_s = [
                np.random.multivariate_normal(np.zeros((d)), np.eye(d))
                for i in range(self.num_actions)
            ]

        # Compute last-layer representation for the current context
        c = context.reshape((1, self.context_dim))
        z_context = self.model_hl.predict(c)

        # Apply Thompson Sampling to last-layer representation
        vals = [
            np.dot(beta_s[i], z_context.T)  for i in range(self.num_actions)
        ]
        return np.argmax(vals)

    def update(self, context, action, reward):
        """Updates action posterior using the linear Bayesian regression formula.

        Args:
        context: Last observed context.
        action: Last observed action.
        reward: Last observed reward.
        """

        self.step += 1
        # Update data buffer
        self.data_h.add(context, action, reward)

        # Update latent data buffer
        c = context.reshape((1, self.context_dim))
        z_context = self.model_hl.predict(c)
        self.latent_h.add(z_context, action, reward)

        self.actions.append(action)
        self.rewards.append(reward)

        self.regret += self.bandit.best_reward(context) - reward
        self.regrets.append(self.regret)

        
        # Retrain the network on the original data (data_h)
        if self.step % self.update_freq_nn == 0:
            for _ in range(self.training_epochs):
                x, y = self.data_h.get_batch(self.batch_size)
                self.model.fit(x, y, epochs=1, verbose=0)

            # Update the latent representation of every datapoint collected so far
            new_z = self.model_hl.predict(self.data_h.contexts)
            self.latent_h.replace_data(contexts=new_z)


        # Update the Bayesian Linear Regression
        if self.step % self.update_freq_lr == 0:
            # Find all the actions to update
            actions_to_update = self.latent_h.actions[:-self.update_freq_lr]

            for action_v in np.unique(actions_to_update):

                # Update action posterior with formulas: \beta | z,y ~ N(mu_q, cov_q)
                z, y = self.latent_h.get_data(action_v)

                # The algorithm could be improved with sequential formulas (cheaper)
                s = np.dot(z.T, z)

                # Some terms are removed as we assume prior mu_0 = 0.
                precision_a = s + self.lambda_prior * np.eye(self.latent_dim)
                cov_a = np.linalg.inv(precision_a)
                mu_a = np.dot(cov_a, np.dot(z.T, y))

                # Inverse Gamma posterior update
                a_post = self.a0 + z.shape[0] / 2.0
                b_upd = 0.5 * np.dot(y.T, y)
                b_upd -= 0.5 * np.dot(mu_a.T, np.dot(precision_a, mu_a))
                b_post = self.b0 + b_upd

                # Store new posterior distributions
                self.mu[action_v] = mu_a
                self.cov[action_v] = cov_a
                self.precision[action_v] = precision_a
                self.a[action_v] = a_post
                self.b[action_v] = b_post




