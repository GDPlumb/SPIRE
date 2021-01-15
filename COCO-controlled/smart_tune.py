from pulp import LpMinimize, LpProblem, LpStatus, lpSum, LpVariable

def get_intervention(B, M, S, N):
    model = LpProblem(name="small-problem", sense=LpMinimize)
    x4 = LpVariable(name="B2M", lowBound=0)
    x5 = LpVariable(name="M2B", lowBound=0)
    x6 = LpVariable(name="S2N", lowBound=0)
    x7 = LpVariable(name="N2S", lowBound=0)
    x8 = LpVariable(name="M2N", lowBound=0)
    x9 = LpVariable(name="N2M", lowBound=0)
    x10 = LpVariable(name="B2S", lowBound=0)
    x11 = LpVariable(name="S2B", lowBound=0)
    x12 = LpVariable(name="M2M", lowBound=0)
    x13 = LpVariable(name="N2N", lowBound=0)
    x14 = LpVariable(name="B2B", lowBound=0)
    x15 = LpVariable(name="S2S", lowBound=0)
    x16 = LpVariable(name="B2N", lowBound=0)
    x17 = LpVariable(name="N2B", lowBound=0)
    model += (B + x5 + x11 + x14 + x17 == M + x4 + x9 + x12, "B = M")
    model += (B + x5 + x11 + x14 + x17 == S + x10 + x7 + x15, "B = S")
    model += (B + x5 + x11 + x14 + x17 == N + x8 + x6 + x13 + x16, "B = N")
    # model += (x4 + x12  + x14 == x6 + x8 + x10 + x13 + x15 + x16, "box")
    model += (x5 + x9 + x11 + x17 == x7, "add")
    # model += (x4 <=  B, "p4 < 1")
    # model += (x5 <=  M, "p5 < 1")
    # model += (x6 <=  S, "p6 < 1")
    # model += (x7 <=  N, "p7 < 1")
    # model += (x8 <=  M, "p8 < 1")
    # model += (x9 <=  N, "p9 < 1")
    # model += (x10 <= B, "p10 < 1")
    # model += (x11 <= S, "p11 < 1")
    # model += (x12 <= M, "p12 < 1")
    # model += (x13 <= N, "p13 < 1")
    # model += (x14 <= B, "p14 < 1")
    # model += (x15 <= S, "p15 < 1")
    # model += (x16 <= B, "p16 < 1")
    # model += (x17 <= N, "p17 < 1")
    model += (x16 == 0, "removing p16, haven't implemented it yet")
    model += (x17 == 0, "removing p17, haven't implemented it yet")
    # model += x4+ 1.5 * x5 + x6 + 1.5* x7 + x8 + 1.5 * x9 + x10 + 1.5 * x11 + 2 * x12 + 2 * x13 + 2 * x14 + 2 * x15 + 3 * x16 + 3 * x17
    # model += (x4 + x12  + x14 - x6 - x8 - x10 - x13 - x15 - x16) * 100 +  x4+ 1.5 * x5 + x6 + 1.5* x7 + x8 + 1.5 * x9 + x10 + 1.5 * x11 + 2 * x12 + 2 * x13 + 2 * x14 + 2 * x15 + 3 * x16 + 3 * x17
    # minimize box bias, don't care about add bias
    model += (-x4 - x12  - x14 + x6 + x8 + x10 + x13 + x15 + x16) * 100 +  x4+ 1.5 * x5 + x6 + 1.5* x7 + x8 + 1.5 * x9 + x10 + 1.5 * x11 + 3 * x12 + 3 * x13 + 3 * x14 + 3 * x15 + 3 * x16 + 3 * x17
    status = model.solve()
    if status != 1:
        print('ERROR: COULD NOT SOLVE LINEAR PROGRAM')
    initv = {
        'B': B,
        'M': M,
        'S': S,
        'N': N
    }
    interventions = {}
    for var in model.variables():
        if var.value() != 0:
            interventions[var.name] = var.value() / initv[var.name[0]]
#             print(f"{var.name}: {var.value()} p=%.2f" % (var.value() / initv[var.name[0]]))
    return translate_interventions(interventions)
def translate_interventions(interventions):
    names = {}
    for k in ['both', 'just_main', 'just_spurious', 'neither']:
        names[k] = {'orig': 1.0}
    translation = {
        'B2M': ('both', 'spurious-box'),
        'B2S': ('both', 'main-box'),
        'B2B': ('both', 'custom_marco'),
        'M2B': ('just_main', 'just_main+just_spurious'),
        'M2N': ('just_main', 'main-box'),
        'M2M': ('just_main', 'custom_marco'),
        'S2B': ('just_spurious', 'just_spurious+just_main'),
        'S2N': ('just_spurious', 'spurious-box'),
        'S2S': ('just_spurious', 'custom_marco'),
        'N2M': ('neither', 'neither+just_main'),
        'N2S': ('neither', 'neither+just_spurious'),
        'N2N': ('neither', 'custom_marco'),
    }
    for x, v in interventions.items():
        k, a = translation[x]
        if a == 'custom_marco':
            for a in ['%s+fake_main_box' % k, '%s+fake_spurious_box' % k]:
                names[k][a] = v / 2.
            pass
        else:
            names[k][a] = v
    return names
