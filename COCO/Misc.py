
def get_sc_strength(data):
    return data['s_given_m'] - data['s_given_not_m']

def get_cf_score(data):
    return data['cf_score']

def get_d_gap(data):
    v_b = data['both']
    v_m = data['just_main']
    if v_b == -1 or v_m == -1:
        return -1
    else:
        return data['both'] - data['just_main']

def get_h_gap(data):
    v_s = data['just_spurious']
    v_n = data['neither']
    if v_s == -1 or v_n == -1:
        return -1
    else:
        return data['neither'] - data['just_spurious'] 
