
%% Attractor Network Model
% Firing rates of 12 units are simulated over time:
% 1) 2 'intention nodes' representing voluntary colour choice (blue/green)
% 2) 2 'sensory nodes' representing perceptual choice (left/right)
% 3) 4 'action nodes' representing the movement output (target)
% 4) 4 'cost nodes' representing the cost of each action (distance from target)
% Units can excite or inhibit each other to different degrees according
% to the connectivity matrix W in which W(i,j) represents connection
% strength from unit j to unit i.
% An action is initiated when a fixed firing rate threshold is crossed
% Firing rates continue to be updated after action onset (for duration of
% non-decision time), which can cause a Change of Mind

clear all
clc

rand('state', sum(100*clock)); % set seed of randomisation to current time

%% Run n trials
n_trials = 100; % number of trials to simulate

%% Model parameters
col = 48.0; % strength of colour intention
coh = 1.03; % sensory evidence (dot-motion input)
hierarchy = 2.01; % degree of hierarchical control


%% define model parameters & network structure
% set parameters to values obtained by cmaes optimization
W_sensory = 1.50; % initial weight sensory nodes -> action nodes
W_sensory_auto = 0.25; % initial weight for sensory auto-connections
W_intentional = 0.97; % initial weight for intentional nodes -> action nodes
W_cost = -0.97; % initial weight for cost nodes -> action nodes
W_inhibit = -0.52; % initial weight for all inhibitory connections

conflict = 0; % simulate conflict trials? (if 0: simulates test trials)

% define nodes and their weights (strength of connection from j to i)
if ~conflict % Test trials
    W = [ 0 W_inhibit   0   0   0   0   0   0  W_cost/2  W_cost/2   0   0; ...     % 1=I1 (blue)
        W_inhibit   0   0   0   0   0   0   0   0   0  W_cost/2  W_cost/2; ...      % 2=I2 (green)
        0   0 W_sensory_auto W_inhibit   0   0   0   0   0   0   0   0; ...      % 3=S1 (left)
        0   0 W_inhibit W_sensory_auto   0   0   0   0   0   0   0   0;...       % 4=S2 (right)
        W_intentional   0   W_sensory   0   0  W_inhibit*2 W_inhibit W_inhibit  W_cost   0   0   0;... % 5=A1 (blue-left)
        W_intentional   0   0   W_sensory  W_inhibit*2   0 W_inhibit W_inhibit   0  W_cost   0   0;... % 6=A2 (blue-right)
        0  W_intentional   W_sensory   0 W_inhibit W_inhibit   0  W_inhibit*2   0   0  W_cost   0;... % 7=A3 (green-left)
        0  W_intentional   0   W_sensory W_inhibit W_inhibit  W_inhibit*2   0   0   0   0  W_cost;... % 8=A4 (green-right)
        0   0   0   0   0   0   0   0   0   0   0   0;...          % 9=C1 (cost of action blue-left)
        0   0   0   0   0   0   0   0   0   0   0   0;...          %10=C2 (cost of action blue-right)
        0   0   0   0   0   0   0   0   0   0   0   0;...          %11=C3 (cost of action green-left)
        0   0   0   0   0   0   0   0   0   0   0   0];            %12=C4 (cost of action green-right)
elseif conflict % CONFLICT: blue maps onto both left actions, green on both right
    W = [ 0 W_inhibit   0   0   0   0   0   0  W_cost/2  W_cost/2   0   0; ...     % 1=I1 (blue)
        W_inhibit   0   0   0   0   0   0   0   0   0  W_cost/2  W_cost/2; ...      % 2=I2 (green)
        0   0 W_sensory_auto W_inhibit   0   0   0   0   0   0   0   0; ...      % 3=S1 (left)
        0   0 W_inhibit W_sensory_auto   0   0   0   0   0   0   0   0;...       % 4=S2 (right)
        W_intentional   0   W_sensory   0   0  W_inhibit*2 W_inhibit W_inhibit  W_cost   0   0   0;... % 5=A1 (BLUE-LEFT)
        0 W_intentional  0   W_sensory  W_inhibit*2   0 W_inhibit W_inhibit   0  W_cost   0   0;... % 6=A2 (GREEN-RIGHT)
        W_intentional 0   W_sensory   0 W_inhibit W_inhibit   0  W_inhibit*2   0   0  W_cost   0;... % 7=A3 (BLUE-LEFT)
        0  W_intentional   0   W_sensory W_inhibit W_inhibit  W_inhibit*2   0   0   0   0  W_cost;... % 8=A4 (GREEN-RIGHT)
        0   0   0   0   0   0   0   0   0   0   0   0;...          % 9=C1 (cost of action blue-left)
        0   0   0   0   0   0   0   0   0   0   0   0;...          %10=C2 (cost of action blue-right)
        0   0   0   0   0   0   0   0   0   0   0   0;...          %11=C3 (cost of action green-left)
        0   0   0   0   0   0   0   0   0   0   0   0];            %12=C4 (cost of action green-right)
end


Nunits = length(W); % number of units
neural_noise = 2;  % level of neural noise
rmax = 100;         % maximum firing rate
rinit = 10;         % all neurons start with same background firing rate
thresh = 40;        % threshold for action initiation

% input currents
Hz = 60; % strength of input currents
I = [Hz*(1+col/100) Hz*(1-col/100)]; % strengt of blue/green input according to intentional strength
S = [Hz*(1+coh/100) Hz*(1-coh/100)]; % strength of left/right dot motion input according to strength of sensory evidence


%% timing parameters
dt = .001;              % time step in s
tau = .1;          % base time constant in sec

nd_t = [200 180]/(dt*1000);   % non-decision time in ms [sensory motor] delays
tSon = nd_t(1);     % onset of sensory stimulation = sensory nd time

deadline = 1000;        % deadline for response initiation


%% define movement parameters
%target locations (4 targets corresponding to top/bottom left/right)
targets = [-200 -250;... % blue-left (bottom-left)
    200 250;... % blue-right (top-right)
    -200 250;... % green-left (top-left)
    200 -250]; % green-right (bottom-right)

vel = .7*dt*1000;%.7*dt*1000;     % assume constant velocity of 1 pixel/ms to determine cursor position

% action cost -> based on distance to targets
C = [pdist([[0 0]; targets(1,:)],'euclidean'),...%initial cost of action 1 = distance to target 1 etc. -> decreases/increases over time, depending on current position
    pdist([[0 0]; targets(2,:)],'euclidean'),...
    pdist([[0 0]; targets(3,:)],'euclidean'),...
    pdist([[0 0]; targets(4,:)],'euclidean')];

Cnorm = [C/Hz]'; %normalisation factor for costs


%% initialise output variables
choice_initial = nan(n_trials,1); % first action that wins
error_initial = nan(n_trials,2); % accuracy of first action [according to dot motion (left/right), according to colour intention (blue/green)]
choice_final = nan(n_trials,1); % final target that is reached
error_final = nan(n_trials,2); % accuracy of final action [according to dot motion (left/right), according to colour intention (blue/green)]
CoMn = nan(n_trials,1); % Change of Mind (any type, i.e., did switch from one action to another one occur?)
CoMov = nan(n_trials,1); % Change of Movement (i.e., switch between left/right, but no colour switch; e.g., A1 -> A2)
CoMovInt = nan(n_trials,1); % Change of Movement + Intention (i.e.,switch between left/right AND switch between blue/green; e.g., A1 -> A4)
CoInt = nan(n_trials,1); % Change of Intention (switch between blue/green WITHOUT switch between left/right)
RT = nan(n_trials,1); % time of initial response onset
MT = nan(n_trials,1); % movement time = time it took to reach target after action initiation
CoMt = nan(n_trials,1); % time at which CoM occurred (relative to response initiation)
t_fin = nan(n_trials,1); % last time step of a given simulation (used for plotting)


%% initialise network
%Set up time vector
tmax = (deadline + sum(nd_t))*(dt);     % max time in sec
tvec = 0:dt:tmax;       % time vector
Nt = length(tvec);      % number of time steps

%set all firing rates to zero)
Input = zeros(Nunits,Nt,n_trials);
Input(1:2,1,:) = repmat(I',[1,1,n_trials]);
Input(9:12,1,:) = repmat(C./Cnorm',[1,1,n_trials]);

r = rinit*ones(Nunits, Nt,n_trials);
c = repmat(C',[1,Nt, n_trials]);
pos = zeros(2,Nt,n_trials);
sigma_sq = nan(Nunits,Nt,n_trials);

correct_direction_rand = nan;
correct_colour_rand = nan;

for tr = 1:n_trials
    
    % if conflict trials, randomly shuffle colour intention
    if conflict
        I = I(randperm(2));
    end
    
    % if dot coherence/intentional strength is 0, randomly determine correct choice
    if coh == 0; correct_direction_rand = randperm(2); end
    if col == 0; correct_colour_rand = randperm(2); end
    
    
    previous_choice = nan; %keep track of what the previously chosen action was within a trial (in case action changes)
    
    for t=2:Nt
        
        % define inputs into 'intention nodes' (I1/I2) at each time step
        Input(1:2,t,tr) = I'; %+ I_noise*randn(2,1);
        
        % define inputs into 'sensory nodes' (S1/S2) at each time step
        if t<tSon % no sensory evidence before sensory delay
            Input(3:4,t,tr) = 0;
        else % sensory evidence is accumulated over time
            Input(3:4,t,tr) = S'; 
        end
        
        % define inputs into 'cost nodes' (C1 - C4) at each time step
        % action cost = distance of each target from current position
        for a = 1:4
            c(a,t,tr) = pdist([pos(:,t,tr)'; targets(a,:)],'euclidean'); 
        end
        Input(9:12,t,tr) = c(:,t,tr)./Cnorm; % action cost is fed into cost nodes
        
        % update firing rates
        stim = Input(:,t,tr) + W*r(:,t-1,tr); % input to each unit based on weight matrix and external input
        newr = r(:,t-1,tr) + (stim-r(:,t-1,tr))*dt/tau;  % Euler–Maruyama update of rates
        
        
        % add noise according to hierarchical structure
        for i = 1:Nunits
            if i == 5 || i == 6
                sigma_sq(i,t,tr) = neural_noise-hierarchy*(r(1,t-1,tr)/100)*neural_noise; % A1 & A2: level of noise of depends on previous state of Intention 1 (blue)
            elseif i == 7 || i == 8
                sigma_sq(i,t,tr) = neural_noise-hierarchy*(r(2,t-1,tr)/100)*neural_noise; % A3 & A4: level of noise depends on previous state of Intention 2 (green)
            else
                sigma_sq(i,t,tr) = neural_noise; % all other nodes: level of noise is fixed = neural noise
            end
            sigma_sq(i,t,tr) = max(sigma_sq(i,t,tr),0);                 % noise is always positive
            newr(i) = newr(i) + sqrt(sigma_sq(i,t,tr))*randn(1,1); % add noise to firing rate
        end
        
        r(:,t,tr) = max(newr,0);               % rates are always positive
        r(:,t,tr) = min(r(:,t,tr),rmax);        % rates are below rmax
        
        r_sort = sort(r(5:8,t,tr)); %sort firing rates according to size
        % check if any action node has crossed threshold for action initiation
        if max(r(5:8,t,tr)) > thresh && diff(r_sort(3:4)) > 10  % at least 10 Hz difference between 2 strongest nodes
            [m ind] = max(r(5:8,t,tr)); % find winning action to determine movement direction
            
            % generate movement trajectory
            % (lagging behind current firing rate according to motor delay)
            move_dir=(targets(ind,:)'-pos(:,t+nd_t(2),tr))./norm(targets(ind,:)'-(pos(:,t+nd_t(2),tr))); % determine direction of movement based on chosen target
            pos(:,t+1+nd_t(2),tr) = pos(:,t+nd_t(2),tr) + vel*move_dir; % determine future position of cursor depending on movement direction and velocity
            
            %first time action threshold is reached, get RT & initial choice
            if isnan(choice_initial(tr))
                RT(tr) = (t*dt*1000); % RT in ms
                % stop trial if response deadline is exceeded
                if (RT(tr)+nd_t(2)) > deadline
                    RT(tr) = nan;
                    break
                end
                
                % if response initiated before stimulus onset, interrupt
                if RT(tr) < tSon
                    break
                end
                
                choice_initial(tr) = ind; % index of winning action
                CoMn(tr) = 0;
                
                
                %check if INITIAL action corresponds to actual sensory input
                %(e.g., left dot-motion direction is stronger right and response = left, or v.v.)
                if (coh == 0 && correct_direction_rand(1) == 1) || (((S(1) > S(2) && (choice_initial(tr) == 1 || choice_initial(tr) == 3)) || (S(2) > S(1) && (choice_initial(tr) == 2 || choice_initial(tr) == 4))))
                    error_initial(tr,1) = 0;
                else
                    error_initial(tr,1) = 1;
                end
                
                %check if INITIAL action corresponds to actual colour intention
                %(e.g., blue stronger than green and response = blue, or v.v.)
                if (col == 0 && correct_colour_rand(1) == 1) || ((I(1) > I(2) && choice_initial(tr) <= 2) || (I(2) > I(1) && choice_initial(tr) > 2))
                    error_initial(tr,2) = 0;
                else
                    error_initial(tr,2) = 1;
                end
                
            else % if action threhsold crossed 2nd/3rd... time, check if it is the same action that had crossed threshold previously or whether it's a different action (only latter case = Change of Mind)
                if ind ~= previous_choice
                    CoMn(tr) = CoMn(tr) + 1; % count how often it happens that different action crosses threshold
                    if CoMn(tr) == 1 % for first Change of Mind: Get time relative to initial action onset
                        CoMt(tr) = (t*dt*1000) - RT(tr);
                    end
                end
            end
            previous_choice = ind;
        elseif max(r(5:8,t,tr)) <= thresh && ~isnan(choice_initial(tr)) %if an action had previously crossed threshold -> continue with same action (even if currently no action is above threshold) [ensures continuity of movement]
            pos(:,t+1+nd_t(2),tr) = pos(:,t+nd_t(2),tr) + move_dir*vel;%*r(ind,t,tr)/thresh; % velocity is relative to how much above/below threshold selected action node is
        else % if no action had previously crossed threshold, cursor position = previous position
            pos(:,t+1+nd_t(2),tr) = pos(:,t+nd_t(2),tr);
        end
        
        
        %% end of trial
        % stop after non-decision time has passed (or if target has been reached)
        % -> get final response and stop simulation
        if (t > (RT(tr)/(dt*1000) + sum(nd_t))) || (~isempty(find(abs(pos(1,t+1+nd_t(2),tr))' > abs(targets(:,1)))) && ~isempty((abs(pos(2,t+1+nd_t(2),tr))' > abs(targets(:,2)))))
            choice_final(tr) = ind;
            
            %check if FINAL action corresponds to actual sensory input
            %(e.g., left dot-motion direction is stronger right and response = left, or v.v.)
            if (S(1) > S(2) && (choice_final(tr) == 1 || choice_final(tr) == 3)) || (S(2) > S(1) && (choice_final(tr) == 2 || choice_final(tr) == 4)) || (coh == 0 && correct_direction_rand(1) == 1)
                error_final(tr,1) = 0;
            else
                error_final(tr,1) = 1;
            end
            
            %check if FINAL action corresponds to actual colour intention
            %(e.g., blue stronger than green and response = blue, or v.v.)
            if (I(1) > I(2) && choice_final(tr) < 3) || (I(2) > I(1) && choice_final(tr) > 2) || (col == 0 && correct_colour_rand(1) == 1)
                error_final(tr,2) = 0;
            else
                error_final(tr,2) = 1;
            end
            
            % complete remaining movement trajectory & get movement time (time from movement initiation to target hit)
            move_dir=(targets(ind,:)'-pos(:,t+nd_t(2),tr))./norm(targets(ind,:)'-(pos(:,t+nd_t(2),tr))); % determine direction of movement based on chosen target
            while 1
                t = t + 1;
                pos(:,t+1+nd_t(2),tr) = pos(:,t+nd_t(2),tr) + vel*move_dir; % determine new position of cursor depending on movement direction and velocity
                if ~isempty(find(abs(pos(1,t+1+nd_t(2),tr))' > abs(targets(:,1))-5)) && ~isempty((abs(pos(2,t+1+nd_t(2),tr))' > abs(targets(:,2))-5))
                    % get movement time (time from movement initiation to target hit)
                    MT(tr) = ((t)*dt*1000) - (RT(tr));
                    break
                end
            end
            
            
            break
        end
        
        t_fin(tr) = t; % keep track of final t = current t
        
    end
    
    % did a Change of Mind occur? If yes, what type of Change of Mind?
    % only consider single CoM
    if (choice_initial(tr) ~= choice_final(tr)) && CoMn(tr) == 1
        if (choice_initial(tr) == 1 && choice_final(tr) == 2) || (choice_initial(tr) == 2 && choice_final(tr) == 1) || (choice_initial(tr) == 3 && choice_final(tr) == 4) || (choice_initial(tr) == 4 && choice_final(tr) == 3)
            CoMov(tr) = 1;
        elseif ((choice_initial(tr) == 1 && choice_final(tr) == 4) || (choice_initial(tr) == 2 && choice_final(tr) == 3) || ...
                (choice_initial(tr) == 4 && choice_final(tr) == 1) ||  (choice_initial(tr) == 3 && choice_final(tr) == 2))
            CoMovInt(tr) = 1;
        elseif  ((choice_initial(tr) == 1 && choice_final(tr) == 3) || (choice_initial(tr) == 2 && choice_final(tr) == 4) || ...
                (choice_initial(tr) == 3 && choice_final(tr) == 1) || (choice_initial(tr) == 4 && choice_final(tr) == 2))
            CoInt(tr) = 1;
        end
    else
        CoMov(tr) = 0;
        CoMovInt(tr) = 0;
        CoInt(tr) = 0;
    end
end

%% get results (ignore NaN trials)
results.earlyResp = 100*(length(find(RT<tSon))/n_trials); % 3) early response initiation (before stimulus reaches network)
results.missResp = 100*(length(find(isnan(RT))))/n_trials; % 4) % misses (response initiation > 1000 ms)
RT(RT<tSon) = nan; %exclude early RTs from RTs

% for remaining outcome variables, ignore responses that were initiated too early/late
results.RT = nanmean(RT+nd_t(2)); % 1) mean RT in ms
results.MT = nanmean(MT); % 2) mean MT in ms
results.initPercErr = 100*(nansum(error_initial(:,1))/length(find(isnan(RT)==0))); % 5) % initial error perceptual
results.initColErr = 100*(nansum(error_initial(:,2))/length(find(isnan(RT)==0))); % 6) % initial error colour
results.finPercErr = 100*(nansum(error_final(:,1))/length(find(isnan(RT)==0))); % 7) % final error perceptual
results.CoM = 100*(length(find(CoMn == 1))/length(find(isnan(RT)==0))); % 8) % single CoM (any type, including all errors)
results.CoMdouble = 100*(length(find(CoMn > 1))/length(find(isnan(RT)==0))); % 14) % double CoM
results.CoMov = 100*(length(find(CoMov==1))/length(find(isnan(RT)==0))); %15) % all CoMov
results.CoMovInt = 100*(length(find(CoMovInt==1))/length(find(isnan(RT)==0))); %20) % all CoMovInt
results.CoMvert = 100*(length(find(CoInt==1))/length(find(isnan(RT)==0))); %26) % vertical CoM (colour switch)

results

%% plot single-trial outcomes
% which trials to plot? (e.g., plot examples of trials with Change of Mind)

% trialIDs = find(CoMov==1);
trialIDs = [find(CoMov==1); find(CoMovInt==1); find(CoInt == 1); find(CoMn == 0,200)];

plotTrials;    % plots simulated inputs and firing rates over time
