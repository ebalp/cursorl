import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta

from Day1.bandits import BernoulliBandit
from Day1.solvers import Solver, EpsilonGreedy, UCB, BayesianUCB, ThompsonSampling


def plot_results(solvers, solver_names):
    """
    Plot the results by multi-armed bandit solvers.

    Args:
        solvers (list<Solver>): All of them should have been fitted.
        solver_names (list<str>)
    """
    assert len(solvers) == len(solver_names)
    assert all(map(lambda s: isinstance(s, Solver), solvers))
    assert all(map(lambda s: len(s.regrets) > 0, solvers))

    b = solvers[0].bandit

    fig = plt.figure(figsize=(14, 4))
    fig.subplots_adjust(bottom=0.3, wspace=0.3)

    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    # Sub.fig. 1: Regrets in time.
    for i, s in enumerate(solvers):
        ax1.plot(range(len(s.regrets)), s.regrets, label=solver_names[i])

    ax1.set_xlabel('Time step')
    ax1.set_ylabel('Cumulative regret')
    ax1.legend(loc=9, bbox_to_anchor=(1.82, -0.25), ncol=5)
    ax1.grid('k', ls='--', alpha=0.3)

    # Sub.fig. 2: Probabilities estimated by solvers.
    sorted_indices = range(b.n)
    ax2.plot(range(b.n), [b.probas[x] for x in sorted_indices], 'k--', markersize=12, label='True Prob')
    for s in solvers:
        ax2.plot(range(b.n), [s.estimated_probas[x] for x in sorted_indices], 'x', markeredgewidth=2)
    
    ticks = range(solvers[0].bandit.n)
    ax2.set_xticks(ticks, minor=False)
    ax2.set_xticklabels(['a_{}'.format(i) for i in ticks], fontdict=None, minor=False)
    ax2.set_xlabel('Actions')
    ax2.set_ylabel('Estimated')
    ax2.grid('k', ls='--', alpha=0.3)
    ax2.legend(loc='best')

    # Sub.fig. 3: Action counts
    for s in solvers:
        ax3.plot(range(b.n), np.array(s.counts) / float(len(solvers[0].regrets)), ls='steps', lw=2)
    
    ticks = range(solvers[0].bandit.n)
    ax3.set_xticks(ticks, minor=False)
    ax3.set_xticklabels(['a_{}'.format(i) for i in ticks], fontdict=None, minor=False)
    ax3.set_xlabel('Actions')
    ax3.set_ylabel('Frac. # trials')
    ax3.grid('k', ls='--', alpha=0.3)


def plot_beta(bayesian_solver, save=False):
    """
    Plots the pdf's of the distributions of the bandits.
    """ 

    _, ax = plt.subplots(1,1, figsize=(12.5,5))
    x = np.linspace(0,1, 200)
    for i in range(bayesian_solver.bandit.n):
        rv = beta(bayesian_solver._as[i], bayesian_solver._bs[i])
        y = rv.pdf(x)
        line = ax.plot(x, y, lw = 1, label="Bandit : {}".format(i))
        ax.fill_between(x, 0, y, alpha = 0.2, color = line[0].get_color())
    ax.set_ylim(0)
    ax.set_title('Bandits distribution')
    plt.legend(loc = 'best', title=r"Bandits")
    if save:
        plt.savefig('bandits.png', transparent=True)