import numpy as np
import sys



#-------------- PARAMETERS --------------#

# Model parameters
dtheta = np.pi/10
N_rotations = 10
phi = 0 

N_time_steps = N_rotations

# Simulation input param
seed = 0
in_state_noise, stoch_env = 0,0
meas_mode_int = 0

# action dictionary: encodes action values as strings from which RL states are built
allowed_states=np.sort([0,1])
allowed_states=np.sort([0,1])
as_dict={0:bytes('-','utf-8'),1:bytes('+','utf-8')}


# Model specifics
model_type = 'quantum'
model_name = 'qbits_SFQ'




#-------------- CREATE PROPER PARAMETERS' DICTIONARIES --------------#

# set noise level in initial state
if model_type=='quantum': eta=0.31
elif model_type=='classical': eta=0.1

if N_time_steps != 0: stoch_lev = 1.0/N_time_steps
else: stoch_levs=0
    
# define model params
params_model=dict( 
    L=1, # size of the spin chain
    basis_kwargs=dict(pauli=False), # basis arguments
    delta_theta = dtheta,
    delay_phase = phi,
    ### general
    N_time_steps=N_time_steps, # evolution duration
    allowed_states=allowed_states,
    measure_mode=['deterministic', 'stochastic'][meas_mode_int],
    initial_state_noise=in_state_noise, # noisy initial state
    noise_level=eta, # noise in initial state is sqrt(noise_level): psi <-- psi + noise_level*random_perturbation
    stochastic_env=stoch_env, # stochastic actions
    stochasticity_level=stoch_lev, # probability to take random action
    model_type='quantum' #model_type, # classical vs quantum model
    )


# define tabular params
params_tabular=dict(
    N_time_steps=N_rotations,
    N_actions=len(allowed_states),
    )

# define Q-learning params
params_QL=dict(
    N_episodes= 4000, # 2*10**4, # train agent
    N_greedy_episodes=100, # test agent: episodes after learning is over
    test_model_params=dict(initial_state_noise=in_state_noise,stochastic_env=stoch_env), # model parameters for test stage
    N_repeat=50, # repetition episodes (for quantum measurements only)
    replay_frequency=40, # every so many episodes replays are performed 
    N_replay=40, # replay episodes to facilitate learning
    alpha=0.1, # learning rate
    lmbda=0.6, # TD(lambda)
    beta_RL_i=10, # initial inverse RL temperature (annealed eps-greedy exploration!)
    beta_RL_f=50, # final inverse RL temperature (annealed eps-greedy exploration!)
    action_state_dict=as_dict, # actions string precision
    exploration_schedule='exp', # exploration schedule
    )


#-------------- PARAMETERS --------------#

model_data= '_dtheta=%0.2f_phi=%0.2f_L=%i' % (params_model['delta_theta'],params_model['delay_phase'],params_model['L'])

save_tuple=(seed,int(params_model['stochastic_env']),int(params_model['initial_state_noise']),params_QL['N_episodes'],N_rotations)
fname = 'QL_data-'+model_type+'_qbits-'+params_model['measure_mode']+'-seed=%i_stochenv_%i_noise_%i_Nepisodes=%i_Nrotations=%s'%save_tuple+model_data+'.pkl'

my_dir='/scratch/lcorreal/qcontrol/RL_bukov/reinforcement_learning/Q_learning'

if model_name !='': my_dir += "/{0}_data".format(model_name)	
else : my_dir += "/data"

# check if directory exists and create it if not
initial_state_noise = params_QL['test_model_params']['initial_state_noise']
stochastic_env=params_QL['test_model_params']['stochastic_env']

save_dir = my_dir+"/test_noise-{0:d}_stochenv-{1:d}".format(initial_state_noise,stochastic_env)