
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from problem_density import LowDensity

import sys
sys.path.insert(0, "../Code/")
from train import train_eval

def eval(sess, pred, X):

        problem = LowDensity()
    
        # Visualize the MSE across the two dimensions
        values = np.linspace(0, 1, 100)
        grid = np.zeros((10000, 2))
        c = 0
        for i in range(100):
            for j in range(100):
                grid[c, 0] = values[i]
                grid[c, 1] = values[j]
                c += 1

        grid_pred = sess.run(pred, feed_dict = {X: grid})
        grid_y = problem.label(grid, noise = 0.0)
        grid_error = np.transpose(np.reshape(grid_pred - grid_y, (100, 100)))

        plt.imshow(grid_error)
        plt.title("Difference between predicted and true value")
        plt.xticks([], [])
        plt.yticks([], [])
        plt.xlabel("Feature 0")
        plt.ylabel("Feature 1")
        plt.colorbar()
            
        plt.savefig("out.pdf")

        plt.close()

        # Evaluate whether or not the heuristic was actually enforced
        diffs = np.zeros((3))
        for i in range(10000):
            x_cur = grid[i, :]
            x_pred = sess.run(pred, feed_dict = {X: np.reshape(x_cur, (1,2))})
            
            c = 0
            
            # Evaluate invariance:  MSE of uniform perturbation with range 0.1
            x_pert = np.copy(x_cur)
            x_pert[1] += np.random.uniform(low = -0.1, high = 0.1)
            x_pred_pert = sess.run(pred, feed_dict = {X: np.reshape(x_pert, (1,2))})

            diffs[c] += (x_pred - x_pred_pert)**2
            c += 1
            
            # Evaluate monotonicity:  Average increase of value after increasing feature by 0.05
            for index in [0, 1]:
                x_pert = np.copy(x_cur)
                x_pert[index] += 0.05
                
                x_pred_pert = sess.run(pred, feed_dict = {X: np.reshape(x_pert, (1,2))})
                
                diffs[c] += x_pred_pert - x_pred
                c += 1

        diffs /= 10000

        out = {}
        out["Model MSE"] = np.mean(grid_error ** 2)
        out["MSE of perturbing Feature 1"] = diffs[0]
        out["Mean Increase of increasing Feature 0"] = diffs[1]
        out["Mean Increase of increasing Feature 1"] = diffs[2]
        with open("tests.txt", "w") as outfile:
            json.dump(out, outfile)

problem = LowDensity()
x = problem.gen(100)
y = problem.label(x)

train_eval(x, y, "regression", eval_func = eval)

train_eval(x, y, "regression", eval_func = eval, heuristics = [["inv", 1, 0.1, 1000.0]])

train_eval(x, y, "regression", eval_func = eval, heuristics = [["mon", 0, 0.1, 0.1, 1.0]])

train_eval(x, y, "regression", eval_func = eval, heuristics = [["mon", 0, 0.1, 0.1, 1.0], ["inv", 1, 0.1, 1000.0]])

train_eval(x, y, "regression", eval_func = eval, heuristics = [["mon", 0, 0.1, 1.0, 1.0], ["inv", 1, 0.1, 1000.0]])

train_eval(x, y, "regression", eval_func = eval, heuristics = [["mon", 0, 0.1, 10.0, 1.0], ["inv", 1, 0.1, 1000.0]])


