
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from problem_logical import Logical

import sys
sys.path.insert(0, "../Code/")
from train import train_eval

def eval(sess, pred, X, n = 500):

        problem = Logical()
    
        # Plot the learned function for dim 0 and 1
        acc = np.zeros((3))
        c = 1
        for mode in ["bad", "random", "zeros"]:
        
            if mode == "bad":
                x = problem.gen_bad(n)
            elif mode == "random":
                x = problem.gen_random(n)
            elif mode == "zeros":
                x = problem.gen_zeros(n)

            y_hat = sess.run(pred, feed_dict = {X: x})

            indices_0 = np.where(y_hat < 0.5)[0]
            indices_1 = np.where(y_hat >= 0.5)[0]

            acc[c - 1] = np.mean(problem.label(x) == np.array(y_hat >= 0.5))

            plt.subplot(3, 1, c)
            plt.scatter(x[indices_0, 0], x[indices_0, 1], marker='x')
            plt.scatter(x[indices_1, 0], x[indices_1, 1], marker='+')
            plt.xlabel("Feature 0")
            plt.ylabel("Feature 1")
            plt.title("Features 2 and 3 drawn using: " + mode)
            
            c += 1
    
        plt.tight_layout()

        plt.savefig("out.pdf")

        plt.close()

        # Evaluate whether or not the heuristic was actually enforced on new data
        x = problem.gen_bad(n)
        diffs = np.zeros((6))
        for i in range(x.shape[0]):
            x_cur = x[i, :]
        
            x_pred = sess.run(pred, feed_dict = {X: np.reshape(x_cur, (1,4))})
            
            c = 0
            
            # Evaluate invariance:  MSE of uniform perturbation with range 0.1
            for indices in [[2], [3], [2,3]]:
            
                x_pert = np.copy(x_cur)
                for i in indices:
                    x_pert[i] += np.random.uniform(low = -0.1, high = 0.1)

                x_pred_pert = sess.run(pred, feed_dict = {X: np.reshape(x_pert, (1,4))})

                diffs[c] += (x_pred - x_pred_pert)**2
                c += 1
                
            # Evaluate monotonicity:  Average increase of value after increasing feature by 0.05
            for index in [0, 1, 2]:
                x_pert = np.copy(x_cur)
                x_pert[index] += 0.05
                
                x_pred_pert = sess.run(pred, feed_dict = {X: np.reshape(x_pert, (1,4))})
                
                diffs[c] += x_pred_pert - x_pred
                c += 1
                
        diffs /= n

        out = {}
        out["Model Acc: Bad"] = acc[0]
        out["Model Acc:  Random"] = acc[1]
        out["Model Acc:  Zeros"] = acc[2]
        out["MSE of perturbing Feature 2"] = diffs[0]
        out["MSE of perturbing Feature 2"] = diffs[1]
        out["MSE of perturbing Features 2 and 3"] = diffs[2]
        out["Mean Increase of increasing Feature 0"] = diffs[3]
        out["Mean Increase of increasing Feature 1"] = diffs[4]
        out["Mean Increase of increasing Feature 2"] = diffs[5]
        with open("tests.txt", "w") as outfile:
            json.dump(out, outfile)

problem = Logical()
x = problem.gen_bad(200)
y = problem.label(x)

train_eval(x, y, "binary_classification", eval_func = eval)

# Show the effects of discouraging the use of one or both of the 'bad' features
train_eval(x, y, "binary_classification", eval_func = eval, heuristics = [["inv", 2, 0.1, 1000.0]])
train_eval(x, y, "binary_classification", eval_func = eval, heuristics = [["inv", 2, 0.1, 1000.0], ["inv", 3, 0.1, 1000.0]])

# Show that the monotonicity constraints can be without harming learning
train_eval(x, y, "binary_classification", eval_func = eval, heuristics = [["mon", 0, 0.1, 1.0, 1.0]])
train_eval(x, y, "binary_classification", eval_func = eval, heuristics = [["mon", 0, 0.1, 1.0, 1.0], ["mon", 1, 0.1, 1.0, 1.0]])

# Show that the monotonicity constraints can be used to harm learning (to verify that they work)
train_eval(x, y, "binary_classification", eval_func = eval, heuristics = [["mon", 0, 0.1, 0.1, -1.0], ["mon", 2, 0.1, 0.1, -1.0]])
train_eval(x, y, "binary_classification", eval_func = eval, heuristics = [["mon", 0, 0.1, 1.0, -1.0], ["mon", 2, 0.1, 1.0, -1.0]])
