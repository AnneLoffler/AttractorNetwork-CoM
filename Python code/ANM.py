import numpy as np
import time, os

""" ## Attractor Network Model ##
Firing rates of 12 units are simulated over time:
1) 2 'intention nodes' representing voluntary colour choice (blue/green)
2) 2 'sensory nodes' representing perceptual choice (left/right)
3) 4 'action nodes' representing the movement output (target)
4) 4 'cost nodes' representing the cost of each action (distance from target)
Units can excite or inhibit each other to different degrees according
to the connectivity matrix W in which W(i,j) represents connection
strength from unit j to unit i.
An action is initiated when a fixed firing rate threshold is crossed
Firing rates continue to be updated after action onset (for duration of
non-decision time), which can cause a Change of Mind
"""
def ANN(x=[], verbose=False, debugging_plots=False, n_trials = 10000):
    initial_time = time.time()

    if len(x) == 8:
        coh = x[0]
        col = x[1]
        hierarchy = x[2]
        W_sensory = x[3]
        W_sensory_auto = x[4]
        W_intentional = x[5]
        W_cost = x[6]
        W_inhibit = x[7]
    else:
        coh = 3.2
        col = 51.0
        hierarchy = 1.0
        W_sensory = 1.5
        W_sensory_auto = 0.25
        W_intentional = 1
        W_cost = -1
        W_inhibit = -0.5

    np.random.seed(int(time.time())) # set seed of randomisation to current time

    ## define network properties
    # define nodes and their weights (strength of connection from j to i)
    W= np.array([[0, W_inhibit, 0,   0,   0,   0,   0,   0,  W_cost/2, W_cost/2,   0,   0],     # 1=I1 (blue)
                [W_inhibit,  0,   0,   0,   0,   0,   0,   0,   0,   0,  W_cost/2,  W_cost/2],      # 2=I2 (green)
                [0,   0, W_sensory_auto, W_inhibit, 0,   0,   0,   0,   0,   0,   0,   0],      # 3=S1 (left)
                [0,   0, W_inhibit, W_sensory_auto, 0,   0,   0,   0,   0,   0,   0,   0],       # 4=S2 (right)
                [W_intentional, 0,   W_sensory,   0,   0,  W_inhibit*2, W_inhibit, W_inhibit,  W_cost,   0,   0,   0], # 5=A1 (blue-left)
                [W_intentional, 0,   0, W_sensory,  W_inhibit*2,   0, W_inhibit, W_inhibit,   0,  W_cost,   0,   0], # 6=A2 (blue-right)
                [0,  W_intentional,   W_sensory,   0, W_inhibit, W_inhibit,   0,  W_inhibit*2,   0,   0,  W_cost,   0], # 7=A3 (green-left)
                [0,  W_intentional,   0,   W_sensory, W_inhibit, W_inhibit,  W_inhibit*2,   0,   0,   0,   0,  W_cost], # 8=A4 (green-right)
                [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],          # 9=C1 (cost of action blue-left)
                [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],          #10=C2 (cost of action blue-right)
                [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],          #11=C3 (cost of action green-left)
                [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]])         #12=C4 (cost of action green-right)

    ## timing parameters
    dt = .001                               # time step in sec
    nd_t = np.array([200, 180])/(dt*1000)   # non-decision time in ms [sensory motor] delays
    deadline = 1000                         # deadline for response initiation
    tau = np.array([.1, .1])                # base time constant in sec for changes of rate (100 ms)

    ## neural parameters
    neural_noise = 2    # level of neural noise
    rmax = 100          # maximum firing rate
    Nunits = len(W)     # number of units to simulate
    rinit = 10          # all neurons start with same background firing rate
    thresh = 40         # threshold for action initiation

    ## strength of input currents
    Hz = 60

    ## Set up Input details
    I = np.array([Hz*(1+col/100), Hz*(1-col/100)]) # strengt of blue/green intention
    S = np.array([Hz*(1+coh/100), Hz*(1-coh/100)]) # strength of left/right dot motion (motion coherence)

    tSon = nd_t[0]     # onset of sensory stimulation = sensory nd time

    ## define movement parameters
    #target locations (4 targets corresponding to top/bottom left/right)
    targets = np.array([[-200, -250],  # blue-left (bottom-left)
                        [ 200,  250],  # blue-right (top-right)
                        [-200,  250],  # green-left (top-left)
                        [ 200, -250]]) # green-right (bottom-right)

    centre = np.array([0,0])

    vel = 0.7*dt*1000 # assume constant velocity of 1 pixel/ms to determine cursor position

    # action cost -> based on distance to targets
    # initial cost of action 1 = distance to target 1 etc. -> decreases/increases over time, depending on current position
    C = np.array([[np.linalg.norm(centre + targets[0,:]), np.linalg.norm(centre - targets[1,:]),
                   np.linalg.norm(centre - targets[2,:]), np.linalg.norm(centre - targets[3,:]) ]])

    Cnorm = C/Hz #normalisation factor for costs

    # Note: We had to define this..
    def mat_repmat(x, m, n, p):
        return np.array([np.tile(x, (m,n)) for i in range(p)]).transpose()

    results = np.full((n_trials, 10), np.nan)

    ## set up output variables
    choice_initial = np.full(n_trials, np.nan)  # first action that wins
    error_initial = np.full((n_trials,2), np.nan) # accuracy of first action [according to dot motion (left/right), according to colour intention (blue/green)]
    choice_final = np.full(n_trials, np.nan)    # final target that is reached
    error_final = np.full((n_trials,2), np.nan) # accuracy of final action [according to dot motion (left/right), according to colour intention (blue/green)]
    CoMn = np.full(n_trials, np.nan)            # Change of Mind (any type, i.e., did switch from one action to another one occur?)
    CoMov = np.full(n_trials, np.nan)           # Change of Movement (i.e., switch between left/right, but no colour switch e.g., A1 -> A2)
    CoMovInt = np.full(n_trials, np.nan)        # Change of Movement + Intention (i.e.,switch between left/right AND switch between blue/green e.g., A1 -> A4)
    CoInt = np.full(n_trials, np.nan)           # Change of Intention (switch between blue/green WITHOUT switch between left/right)
    RT = np.full(n_trials, np.nan)              # time of initial response onset
    MT = np.full(n_trials, np.nan)              # movement time = time it took to reach target after action initiation

    # Not used atm..
    CoMt = np.full(n_trials, np.nan) # time at which CoM occurred (relative to response initiation)
    t_fin = np.full(n_trials, np.nan) # last time step of a given simulation (used for plotting)

    ## initialise network
    #Set up time vector
    tmax = (deadline + sum(nd_t))*dt  # max time in sec
    tvec = np.arange(0,tmax,dt)       # time vector         Note: Normally it was 0,tmax+dt,dt..
    Nt = len(tvec)                    # number of time steps

    #set all firing rates to zero)
    Input = np.zeros((Nunits,Nt,n_trials))
    Input[:2,:1,:] = mat_repmat(I.transpose(),1,1,n_trials)
    Input[8:12,:1,:] = mat_repmat((C/Cnorm),1,1,n_trials)

    r = rinit*np.ones((Nunits, Nt,n_trials))
    c = np.ones((C.shape[1], Nt, n_trials))*C[0,0]
    #pos = np.zeros((2,Nt,n_trials)) # Note: This was not enough space for most runs
    pos = np.zeros((2,Nt*10,n_trials))
    sigma_sq = np.full((Nunits,Nt,n_trials), np.nan)
    newr = np.full(Nunits, np.nan) # temp array

    correct_direction_rand = 0
    correct_colour_rand = 0


    for tr in range(n_trials):
        #print('=================== trial',tr)
        # if coherence/intentional strength is 0, randomly determine correct choice
        if coh == 0:
            correct_direction_rand = np.random.permutation(2)
        if col == 0:
            correct_colour_rand = np.random.permutation(2)

        previous_choice = np.nan #keep track of what the previously chosen action was within a trial (in case action changes)

        for t in range(1,Nt): # We've already set the states for t=0

            # define inputs into 'intention nodes' (I1/I2) at each time step
            Input[:2,t,tr] = I.transpose()    #+ I_noise*randn(2,1)

            # define inputs into 'sensory nodes' (S1/S2) at each time step
            if t < tSon: # no sensory evidence before it is switched on
                Input[2:4,t,tr] = 0
            else: # sensory evidence is accumulated over time
                Input[2:4,t,tr] = S.transpose() # + S_noise*randn(2,1)

            # define inputs into 'cost nodes' (C1 - C4) at each time step
            # action cost = distance of each target from current position
            for a in range(4):
                c[a,t,tr] = np.linalg.norm(pos[:,t,tr] - targets[a]) #+ C_noise*randn(1,1)

            Input[8:12,t,tr] = c[:,t,tr]/Cnorm # action cost is fed into cost nodes

            # update firing rates
            stim = Input[:, t, tr] + W.dot(r[:, t-1, tr]) # input to each unit based on weight matrix and external input
            newr[:8] = r[:8,t-1,tr] + (stim[:8]-r[:8,t-1,tr])*dt/tau[0]  # Euler-Maruyama update of rates
            newr[8:12] = r[8:12,t-1,tr] + (stim[8:12]-r[8:12,t-1,tr])*dt/tau[1] # different tau for cost nodes

            # add noise according to hierarchical structure
            for i in range(Nunits):
                if i == 4 or i == 5:
                    sigma_sq[i,t,tr] = neural_noise-hierarchy*(r[0,t-1,tr]/100)*neural_noise # A1 & A2: level of noise of depends on previous state of Intention 1 (blue)
                elif i == 6 or i == 7:
                    sigma_sq[i,t,tr] = neural_noise-hierarchy*(r[1,t-1,tr]/100)*neural_noise # A3 & A4: level of noise depends on previous state of Intention 2 (green)
                else:
                    sigma_sq[i,t,tr] = neural_noise # all other nodes: level of noise is fixed = neural noise

                sigma_sq[i,t,tr] = max(sigma_sq[i,t,tr],0)              # noise is always positive
                newr[i] += np.sqrt(sigma_sq[i,t,tr])*np.random.randn()  # add noise to firing rate

            r[:,t,tr] = np.maximum(newr,np.zeros(Nunits)) # rates are always positive
            r[:,t,tr] = np.minimum(r[:,t,tr],rmax)        # rates are below rmax

            r_sort = np.sort(r[4:8,t,tr]) #sort firing rates according to size
            # check if any action node has crossed threshold for action initiation
            if max(r[4:8,t,tr]) > thresh and np.diff(r_sort[2:4]) > 10:  # at least 10 Hz difference between 2 strongest nodes
                m = max(r[4:8,t,tr]) # find winning action to determine movement direction
                ind = np.argmax(r[4:8,t,tr])

                # generate movement trajectory
                # (lagging behind current firing rate according to motor delay)
                move_dir = (targets[ind,:]-pos[:,int(t+nd_t[1]),tr])/np.linalg.norm( targets[ind,:]-(pos[:,int(t+nd_t[1]),tr]) ) # determine direction of movement based on chosen target
                try:
                    pos[:,int(t+1+nd_t[1]),tr] = pos[:,int(t+nd_t[1]),tr] + vel*move_dir # determine future position of cursor depending on movement direction and velocity
                except:
                    print('Erorr A:',t+1+nd_t[1], t+nd_t[1], tr)
                    exit()

                #first time action threshold is reached, get RT & initial choice
                if np.isnan(choice_initial[tr]):
                    RT[tr] = (t*dt*1000) # RT in ms

                    # if response initiated before stimulus onset, interrupt
                    if RT[tr] < tSon:
                        break

                    choice_initial[tr] = ind # index of winning action
                    CoMn[tr] = 0


                    #check if INITIAL action corresponds to actual sensory input
                    #(e.g., left dot-motion direction is stronger right and response = left, or v.v.)
                    if (coh == 0 and correct_direction_rand[1] == 1) or\
                       ((S[0] > S[1] and (choice_initial[tr] == 0 or choice_initial[tr] == 2)) or (S[1] > S[0] and (choice_initial[tr] == 1 or choice_initial[tr] == 3))):
                        error_initial[tr,0] = 0
                    else:
                        error_initial[tr,0] = 1

                    #check if INITIAL action corresponds to actual colour intention
                    #(e.g., blue stronger than green and response = blue, or v.v.)
                    if (col == 0 and correct_colour_rand[0] == 1) or\
                       ((I[0] > I[1] and choice_initial[tr] <= 1) or (I[1] > I[0] and choice_initial[tr] > 1)):
                        error_initial[tr,1] = 0
                    else:
                        error_initial[tr,1] = 1

                else: # if action threhsold crossed 2nd/3rd... time, check if it is the same action that had crossed threshold previously or whether it's a different action (only latter case = Change of Mind)
                    if ind != previous_choice:
                        CoMn[tr] += 1 # count how often it happens that different action crosses threshold
                        if CoMn[tr] == 1: # for first Change of Mind: Get time relative to initial action onset
                            CoMt[tr] = (t*dt*1000) - RT[tr]

                previous_choice = ind

            elif max(r[4:8,t,tr]) <= thresh and not np.isnan(choice_initial[tr]): #if an action had previously crossed threshold -> continue with same action (even if currently no action is above threshold) [ensures continuity of movement]
                try:
                    pos[:,int(t+1+nd_t[1]),tr] = pos[:,int(t+nd_t[1]),tr] + move_dir*vel#*r(ind,t,tr)/thresh # velocity is relative to how much above/below threshold selected action node is
                except:
                    print('Error B:',t+1+nd_t[1], t+nd_t[1], tr)
                    exit()

            else: # if no action had previously crossed threshold, cursor position = previous position
                try:
                    pos[:,int(t+1+nd_t[1]),tr] = pos[:,int(t+nd_t[1]),tr]
                except:
                    print('Error C:',t+1+nd_t[1], t+nd_t[1], tr)
                    exit()


            ## end of trial
            # stop after non-decision time has passed (or if target has been reached)
            # -> get final response and stop simulation
            if (t > (RT[tr]/(dt*1000) + sum(nd_t))) or\
               ( np.any(abs(pos[0,int(t+1+nd_t[1]),tr]) > abs(targets[:,0])) and np.any(abs(pos[1,int(t+1+nd_t[1]),tr]) > abs(targets[:,1])) ):
                choice_final[tr] = ind

                #check if FINAL action corresponds to actual sensory input
                #(e.g., left dot-motion direction is stronger right and response = left, or v.v.)
                if (S[0] > S[1] and (choice_final[tr] == 0 or choice_final[tr] == 2)) or\
                   (S[1] > S[0] and (choice_final[tr] == 1 or choice_final[tr] == 3)) or\
                   (coh == 0 and correct_direction_rand[1] == 1):
                    error_final[tr,0] = 0
                else:
                    error_final[tr,0] = 1

                #check if FINAL action corresponds to actual colour intention
                #(e.g., blue stronger than green and response = blue, or v.v.)
                if (I[0] > I[1] and choice_final[tr] < 2) or\
                   (I[1] > I[0] and choice_final[tr] > 1) or\
                   (col == 0 and correct_colour_rand[0] == 1):
                    error_final[tr,1] = 0
                else:
                    error_final[tr,1] = 1

                # draw remaining movement trajectory
                # get movement time (time from movement initiation to target hit)
                move_dir = (targets[ind]-pos[:,int(t+nd_t[1]),tr])/np.linalg.norm(targets[ind]-(pos[:,int(t+nd_t[1]),tr])) # determine direction of movement based on chosen target
                while True:
                    t += 1
                    try:
                        pos[:,int(t+1+nd_t[1]),tr] = pos[:,int(t+nd_t[1]),tr] + vel*move_dir # determine new position of cursor depending on movement direction and velocity
                    except:
                        print('Error D:', t+1+nd_t[1], t+nd_t[1])
                        exit()
                    if np.any(abs(pos[0,int(t+1+nd_t[1]),tr]) > abs(targets[:,0])-5) and np.any(abs(pos[1,int(t+1+nd_t[1]),tr]) > abs(targets[:,1])-5):
                        # get movement time (time from movement initiation to target hit)
                        MT[tr] = (t*dt*1000) - RT[tr]
                        break
                break

            t_fin[tr] = t # keep track of final t = current t

        # did a Change of Mind occur? If yes, what type of Change of Mind?
        # only consider single CoM
        if (choice_initial[tr] != choice_final[tr]) and CoMn[tr] == 1:
            if (choice_initial[tr] == 0 and choice_final[tr] == 1) or\
               (choice_initial[tr] == 1 and choice_final[tr] == 0) or\
               (choice_initial[tr] == 2 and choice_final[tr] == 3) or\
               (choice_initial[tr] == 3 and choice_final[tr] == 2):
                CoMov[tr] = 1
            elif (choice_initial[tr] == 0 and choice_final[tr] == 3) or\
                 (choice_initial[tr] == 1 and choice_final[tr] == 2) or\
                 (choice_initial[tr] == 3 and choice_final[tr] == 0) or\
                 (choice_initial[tr] == 2 and choice_final[tr] == 1):
                CoMovInt[tr] = 1
            elif (choice_initial[tr] == 0 and choice_final[tr] == 2) or\
                 (choice_initial[tr] == 1 and choice_final[tr] == 3) or\
                 (choice_initial[tr] == 2 and choice_final[tr] == 0) or\
                 (choice_initial[tr] == 3 and choice_final[tr] == 1):
                CoInt[tr] = 1
            else:
                print('Is this an error?',tr, choice_initial[tr], choice_final[tr])
        else:
            CoMov[tr] = 0
            CoMovInt[tr] = 0
            CoInt[tr] = 0

    """ # For debugging: Visualization of the paths..
    import matplotlib.pyplot as plt
    for i in range(10):
        plt.plot(pos[0,:,i],pos[1,:,i])
    plt.show()
    """

    ## get results:
    results[:,2] = RT<tSon            # 2) early response initiation (before stimulus reaches network)
    results[:,3] = np.isnan(RT)       # 3) # misses (response initiation > 1000 ms) - find the ones which are nan..!

    #exclude early RTs from RTs
    RT[RT<tSon] = np.nan
    results[:,0] = RT+nd_t[1]         # 0) mean RT in ms
    results[:,1] = RT+nd_t[1]         # 1) mean RT in ms (correct perceptual choices only)
    results[np.nonzero(error_initial[:,0]==1),1] = 0.0

    results[:,4] = error_final[:,0]   # 4) # error perceptual (final)
    results[:,5] = error_initial[:,1] # 5) # error colour (initial)
    results[:,6] = CoMov == 1         # 6) # CoMov
    results[:,7] = CoMovInt == 1      # 7) # CoMovInt
    results[:,8] = CoInt == 1         # 8) # all CoInt
    results[:,9] = CoMn > 1           # 9) # double CoM


    np.set_printoptions(suppress=True)
    statsT = "mean RT(ms)     mean RT(ms)    early resp.      misses        error perc      error colour      CoMov         ChMovInt         CoInt            CoM\n"
    statsT+= "              (corr.choices)   initiation     resp.in>1s       (final)         (initial)                                       (all)          (double)"
    print(statsT)
    if verbose:
        for i in range(len(results)):
            print(round(results[i,0],2), '\t\t',round(results[i,1],2), '\t\t',results[i,2], '\t\t',results[i,3], '\t\t',
                  results[i,4], '\t\t',results[i,5], '\t\t',results[i,6], '\t\t', results[i,7], '\t\t', results[i,8], '\t\t', results[i,9])
    #print(np.round(results,1))

    indices_of_correct_perc_choices = np.nonzero(error_initial[:,0]==0)[0]
    indices_to_keep = np.nonzero(~np.isnan(RT))[0] # where RT is not nan
    np.nonzero(error_initial[:,0]==0)
    stats = "-----"*30 + '\n'

    # Up to here, all results have the same dimensions (=n_trials)
    ## define costs according to DVs of interest
    costs = np.zeros(10)
    costs[0] = np.nanmean(results[:,1]) # All of them - NOTE: NOT USED
    costs[1] = np.nanmean(results[:,1][indices_of_correct_perc_choices]) # costRT
    costs[2] = np.nanmean(results[:,2][indices_to_keep])            # costEarly - minimise absolute rate of too-early responses
    costs[3] = np.nanmean(results[:,3][indices_to_keep])            # costMisses
    costs[4] = np.nanmean(results[:,4][indices_to_keep])            # costPercErr
    costs[5] = np.nanmean(results[:,5][indices_to_keep])            # costColErr
    costs[6] = np.nanmean(results[:,6][indices_to_keep])            # costCoMov
    costs[7] = np.nanmean(results[:,7][indices_to_keep])            # costCoMovInt
    costs[8] = np.nanmean(results[:,8][indices_to_keep])            # costCoInt
    costs[9] = np.nanmean(results[:,9][indices_to_keep])            # costDoubleCoM - minimise absolute rate of double CoM

    costs_std = np.zeros(10)
    costs_std[0] = np.nanstd(results[:,1]) # All of them - NOTE: NOT USED
    costs_std[1] = np.nanstd(results[:,1][indices_of_correct_perc_choices]) # costRT
    costs_std[2] = np.nanstd(results[:,2][indices_to_keep])            # costEarly
    costs_std[3] = np.nanstd(results[:,3][indices_to_keep])            # costMisses
    costs_std[4] = np.nanstd(results[:,4][indices_to_keep])            # costPercErr
    costs_std[5] = np.nanstd(results[:,5][indices_to_keep])            # costColErr
    costs_std[6] = np.nanstd(results[:,6][indices_to_keep])            # costCoMov
    costs_std[7] = np.nanstd(results[:,7][indices_to_keep])            # costCoMovInt
    costs_std[8] = np.nanstd(results[:,8][indices_to_keep])            # costCoInt
    costs_std[9] = np.nanstd(results[:,9][indices_to_keep])            # costDoubleCoM

    for i in range(2,10):
        costs[i] = 100.0*costs[i]
        costs_std[i] = 100.0*costs_std[i]
    stats += ' Average (%):\t '+str(round(costs[1],2))+'    \t '+str(round(costs[2],2))+' \t\t '+str(round(costs[3],2))+\
          '\t\t '+str(round(costs[4],2))+' \t\t '+str(round(costs[5],2))+' \t\t '+str(round(costs[6],2))+\
          '   \t '+str(round(costs[7],2))+'    \t '+str(round(costs[8],2))+'    \t '+str(round(costs[9],2)) + '\n'
    stats += '          SD:\t '+str(round(costs_std[1],2))+'    \t '+str(round(costs_std[2],2))+' \t\t '+str(round(costs_std[3],2))+\
          '\t\t '+str(round(costs_std[4],2))+' \t\t '+str(round(costs_std[5],2))+' \t\t '+str(round(costs_std[6],2))+\
          '   \t '+str(round(costs_std[7],2))+'    \t '+str(round(costs_std[8],2))+'    \t '+str(round(costs_std[9],2)) + '\n'
    stats += "-----"*30 + '\n'
    stats += "     Targets:    570.5           0.0            15.0             100.0-56.6      5.0             5.9             1.7             3.0             0.0\n"
    stats += "-----"*30 + '\n'

    # ---------------- CALCULATE COSTS ----------------
    all_targets = [None, 570.5, 0.0, 15.0, 100.0-56.6, 5.0, 5.9, 1.7, 3.0, 0.0]

    for i in range(1,10):
        if all_targets[i] == 0.0:
            costs[i] = abs(costs[i] - all_targets[i])/100.0
        else:
            costs[i] = abs(costs[i] - all_targets[i])/all_targets[i]

    stats += '       Costs:\t '+str(round(costs[1],2))+'    \t '+str(round(costs[2],2))+' \t\t '+str(round(costs[3],2))+\
          '\t\t '+str(round(costs[4],2))+' \t\t '+str(round(costs[5],2))+' \t\t '+str(round(costs[6],2))+\
          '   \t '+str(round(costs[7],2))+'    \t '+str(round(costs[8],2))+'    \t '+str(round(costs[9],2)) + '\n'
    stats += "-----"*30 + '\n'

    #OVERALL COST
    stats += 'Number of trials: ' + str(n_trials) + ', '
    stats += 'Number of experiments we keept excluding early RTs: '+ str(len(indices_to_keep)) + ', '
    stats += "Number of correct perceptual choices: "+str(len(indices_of_correct_perc_choices)) + '\n'
    stats += "Parameters: " + str(x) + '\n'
    stats += "-----"*30 + '\n'
    stats += 'OVERALL COST: ' + str(sum(costs[1:]))
    stats += ' it took ' + str(round(time.time() - initial_time,2))+' seconds..\n'
    stats += "-----"*30
    print(stats)
    stats = statsT+"\n"+stats


    try:
        filename = 'logs_final/'+str(sum(costs[1:]))
        while os.path.exists(filename+'.log'):
            filename += '_1'
        with open(filename+'.log', 'wt') as logfile:
            logfile.write(stats)
    except:
        pass

    return costs[1], costs[2], costs[3], costs[4], costs[5], costs[6], costs[7], costs[8], costs[9]









if __name__ == '__main__':
    ANN(verbose=True, n_trials=10000)





