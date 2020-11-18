
import numpy as np
import sys

def apply(B, M, S, N, B2M = 0, B2S = 0, M2N = 0, S2N = 0):

    B_new = B
    M_new = M + B2M
    S_new = S + B2S
    N_new = N + M2N + S2N
    
    intervened = B2M + B2S + M2N + S2N
    normalizer = 1 + intervened
    
    B_new /= normalizer
    M_new /= normalizer
    S_new /= normalizer
    N_new /= normalizer
    
    if intervened == 0:
        P_Main_Intervened = -1
    else:
        P_Main_Intervened = (B2S + M2N) / intervened
    
    return B_new, M_new, S_new, N_new, P_Main_Intervened
    
def search(B, M, S, N, steps = 52, cost_metric = "balanced_distribution"):

    B2M_array = np.linspace(0, B, num = steps)
    B2S_array = np.linspace(0, B, num = steps)
    M2N_array = np.linspace(0, M, num = steps)
    S2N_array = np.linspace(0, S, num = steps)

    best = [np.inf]
    for b2m in B2M_array:
        for b2s in B2S_array:
            for m2n in M2N_array:
                for s2n in S2N_array:
                    B_new, M_new, S_new, N_new, P_Main_Intervened = apply(B, M, S, N, B2M = b2m, B2S = b2s, M2N = m2n, S2N = s2n)
                    
                    if cost_metric == "balanced_distribution":
                        cost = (B_new - 0.25)**2 + (M_new - 0.25)**2 + (S_new - 0.25)**2 + (N_new - 0.25)**2
                    else:
                        print("Bad Parameter: ", cost_metric)
                        sys.exit(0)
                    
                    if cost < best[0]:
                        best = [cost, P_Main_Intervened, b2m, b2s, m2n, s2n]
                        
    return best


def mix(p, verbose = True):


    B, M, S, N, P_Main_Intervened = apply(B, M, S, N, B2M = B, B2S = B, M2N = M, S2N = S)

    P_Main = B + M
    P_Spurious = B + S
    P_Main_Spurious = B / (B + S)
    P_Spurious_Main = B / (B + M)

    if verbose:
        print("Both Augmented")
        print(B, M, S, N)
        print("P(Main):", P_Main)
        print("P(Spurious):", P_Spurious)
        print("P(Main|Spurious):", P_Main_Spurious)
        print("P(Spurious|Main):", P_Spurious_Main)
        print("P(Main|Intervened):", P_Main_Intervened)
        print()
        print()
    
    return 0
    
if __name__ == "__main__":
    
    for p in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        print()
        print()
        print("p:", p)
        
        # Given p, calculate the initial weight of each bucket
        B = 0.5 * p
        M = 0.5 * (1 - p)
        S = 0.5 * (1 - p)
        N = 0.5 * p
        
        P_Main = B + M
        P_Spurious = B + S
        P_Main_Spurious = B / (B + S)
        P_Spurious_Main = B / (B + M)
        
        print()
        print("Original Distribution:")
        print(B, S)
        print(M, N)
        print("P(Main):", P_Main)
        print("P(Spurious):", P_Spurious)
        print("P(Main|Spurious):", P_Main_Spurious)
        print("P(Spurious|Main):", P_Spurious_Main)
        
        # Find the best mixture
        
        best = search(B, M, S, N)
        
        b2m = best[2]
        b2s = best[3]
        m2n = best[4]
        s2n = best[5]

        B, M, S, N, P_Main_Intervened = apply(B, M, S, N, B2M = b2m, B2S = b2s, M2N = m2n, S2N = s2n)
        
        P_Main = B + M
        P_Spurious = B + S
        P_Main_Spurious = B / (B + S)
        P_Spurious_Main = B / (B + M)
        
        print()
        print("Mixed Augmentation Distribution:")
        print(B, S)
        print(M, N)
        print("P(Main):", P_Main)
        print("P(Spurious):", P_Spurious)
        print("P(Main|Spurious):", P_Main_Spurious)
        print("P(Spurious|Main):", P_Spurious_Main)
        print("P(Main|Intervened):", P_Main_Intervened)

        print()
        print("Strategy:")
        print(b2m, b2s, m2n, s2n)
