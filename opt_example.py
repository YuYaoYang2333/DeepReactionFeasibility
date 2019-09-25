
#..........net_cv means model function（user-defined）

# BayesOpt
from bayes_opt import BayesianOptimization

def generate_nn(LR, Lstm_hid_dim, D_a, R, C_peanuts, BATCH_SIZE, EPOCHS):
    global accbest
    param = {
        'Lstm_hid_dim' : int(np.around(Lstm_hid_dim)),
        'D_a' : int(np.around(D_a)),
        'R' : int(np.around(R)),
        'BATCH_SIZE' : int(np.around(BATCH_SIZE)),
        'EPOCHS' : int(np.around(EPOCHS)),
         'LR' : max(min(LR, 1), 0.0001),
        'C_peanuts' : max(min(C_peanuts, 0.3), 0.001) } 
    
    print("\nSearch parameters %s" % (param), file = log_file)
    log_file.flush()
    sum_trainloss, sum_trainacc, sum_testloss, sum_testacc = net_cv(**param) # **param important
    
    print("test_loss:", sum_testloss)
    print("test_acc:", sum_testacc)
    
    if (np.max(sum_testacc) > accbest):
        costbest = np.max(sum_testacc)
    return np.max(sum_testacc)

log_file = open('nn-bayesian.log', 'a')
accbest = 0.0
NN_BAYESIAN = BayesianOptimization(generate_nn, 
                              {
                               'Lstm_hid_dim': (32, 256),
                               'LR': (0.0001, 1),
                               'BATCH_SIZE': (1, 128),
                               'D_a': (32, 256),
                               'R': (1, 20),
                               'C_peanuts': (0.001, 0.3),
                               'EPOCHS': (5, 35)
                              }) 
NN_BAYESIAN.maximize(init_points = 10, n_iter = 20)#  acq = 'ei', xi = 0.0, 

#print('Maximum NN accuracy value: %f' % NN_BAYESIAN.res['max'])
print('Best NN parameters: ', NN_BAYESIAN.res)