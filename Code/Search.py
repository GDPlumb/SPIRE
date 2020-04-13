
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
        
def metrics(y, y_hat, verbose = False):
    neg = 0
    pos = 0
    neg_t = 0
    pos_t = 0
    
    
    for i in range(y.shape[0]):
        if y[i] == 0:
            neg += 1
            if y_hat[i] == 0:
                neg_t += 1
        else:
            pos += 1
            if y_hat[i] == 1:
                pos_t += 1
                
    if verbose:
        print("Counts: ", neg_t, neg, pos_t, pos)
                
    if neg > 0:
        TNR = neg_t / neg
    else:
        TNR = 3.14159
    if pos > 0:
        TPR = pos_t / pos
    else:
        TPR = 3.15159
    
    metrics = np.array([TNR, TPR])
    
    return metrics

def search(model, X, y, heuristics, perturber, checker,  learner, use_val = False, X_val = None, y_val = None, min_explainability = 0.8, verbose = False):

    n = X.shape[0]
    covered = np.zeros((n))
    out = []
   
   # Test each of the provided heuristics
    for h in heuristics:
    
        # If we have covered the entire dataset, stop early
        if np.sum(covered) == n:
            if verbose:
                print("\nAll Points Covered")
            break
    
        if verbose:
            print("\nHeuristic: ", h)
        
        # Apply the heuristic and get the model's predictions for those points
        X_pert, y_pert = perturber(model, X, h)
                
        # Determine where the heuristics works
        success = checker(y, y_pert)
        num_succeeded = np.sum(success)
        if verbose:
            print("Success on Train: ", num_succeeded)
        
        # TODO: filter out the points that are outside of the data range after they have been perturbed
        # -  Do we actually want to do this?  Cleaning up the areas around the data distribution may reduce unusual explanations for points near the edge of the distribution
        
        # If this heuristic succeeded for at least some points
        if num_succeeded > 0:
        
            # Update which points have been covered
            covered = np.maximum(covered, success)

            # Check whether or not we can easily summarize where the explanation applies
            region = learner(X, success)
            
            # Check how well the learner did on the training set
            success_hat = region.predict(X)
            m = metrics(success, success_hat, verbose = verbose)
            if verbose:
                print("Train Metrics: ", m)
                
            # If region learned well enough on the Training Data
            if np.all(m > min_explainability):
            
                # If we have Validation Data, check if region generalizes to it
                if use_val:
                    X_val_pert, y_val_pert = perturber(model, X_val, h)
                    success_val = checker(y_val, y_val_pert)
                    success_val_hat = region.predict(X_val)
                    
                    num_succeeded_val = np.sum(success_val)
                    if verbose:
                        print("Success on Val: ", num_succeeded_val)
            
                    m = metrics(success_val, success_val_hat)
                    if verbose:
                        print("Validation Metrics: ", m)
                else:
                    num_succeeded_val = num_succeeded
                    
                # If region learned well enough on all the available data
                if np.all(m > min_explainability) and num_succeeded_val > 0:
                    if verbose:
                        print("Accepted\n")
                    summary = Summary(h, success, region)
                    out.append(summary)
            elif verbose:
                print("Rejected\n")
                
    return out
