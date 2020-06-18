
import matplotlib.pyplot as plt
import numpy as np

def predict(model, im):
    return model(np.expand_dims(im, axis = 0)).numpy() >= 0

def explore(model, X, Y, meta, heuristic, max_display = 3, max_check = 100, show = False):
    
    n = X.shape[0]
    
    count = 0
    
    num = min(n, max_check)
    
    for i in range(num):
        
        im = np.copy(X[i])
        m = meta[i]
        
        p = predict(model, im)
        
        out = heuristic(im, m)
        
        if out is not None:
            
            im_new = out[0]
            p_new = predict(model, im_new)
            
            if p != p_new or show:
                
                print()
                print(i)

                plt.imshow(im)
                plt.show()
                plt.close()

                print("Initial Prediction: ", p)

                plt.imshow(im_new)
                plt.show()
                plt.close()

                print("New Prediction: ", p_new)

                print()
                
                count += 1
                
                if count == max_display:
                    print("Hit max_display")
                    return None
                
    print("Searched ", num, " images")
    return None
