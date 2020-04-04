
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

class Summary:

    def __init__(self, heuristic, indices, region):
        self.heuristic = heuristic
        self.indices = indices
        self.region = region
        
    def print(self):
        print(self.heuristic)
        plt.figure()
        plot_tree(self.region, filled=True) # TODO:  DT specific
        plt.show()

def search(model, X, y, heuristics, perturber, checker,  learner, use_val = False, X_val = None, y_val = None, min_explainability = 0.8, verbose = False):

    n = X.shape[0]
    covered = np.zeros((n))
    out = []
   
   # Test each of the provided heuristics
    for h in heuristics:
    
        # If we have covered the entire dataset, stop early
        if np.sum(covered) == n:
            if verbose:
                print("Stoped Early")
            break
    
        if verbose:
            print(h)
        
        # Apply the heuristic and get the model's predictions for those points
        X_pert, y_pert = perturber(model, X, h)
                
        # Determine where the heuristics works
        success = checker(y, y_pert)
        num_succeeded = np.sum(success)
        if verbose:
            print(num_succeeded)
        
        # TODO: filter out the points that are outside of the data range after they have been perturbed
        # -  Do we actually want to do this?  Cleaning up the areas around the data distribution may reduce unusual explanations for points near the edge of the distribution
        
        # If this heuristic succeeded for at least some points
        if num_succeeded > 0:
        
            # Update which points have been covered
            covered = np.maximum(covered, success)

            # Check whether or not we can easily summarize where the explanation applies
            region = learner(X, success)
            
            # If we were given validation data, use that to check how well region performs
            if use_val:
                X_val_pert, y_val_pert = perturber(model, X_val, h)
                success_eval = checker(y_val, y_val_pert)
                success_eval_hat = region.predict(X_val)
            else:
                success_eval = success
                success_eval_hat = region.predict(X)
            
            # Find the true positive and true negative rates
            m = confusion_matrix(success_eval, success_eval_hat)
            m_d = np.diag(m)
            m_count = np.sum(m, axis = 1)
            metrics = -1.0 * np.ones((m_d.shape[0]))
            for i in range(m_d.shape[0]):
                if m_count[i] != 0.0:
                    metrics[i] = m_d[i] / m_count[i]
                else:
                    metrics[i] = 1.0

            if verbose:
                print(metrics)
            
            # If this group is sufficientely explainable
            if np.all(metrics > min_explainability) and np.sum(success_eval) > 0: # Require that the explanation works at least once on the validation data (if used)
                if verbose:
                    print("Accepted")
                summary = Summary(h, success, region)
                out.append(summary)
            elif verbose:
                print("Rejected")
                
    return out
