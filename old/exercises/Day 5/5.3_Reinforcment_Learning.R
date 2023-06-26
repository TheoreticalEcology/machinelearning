# git clone https://github.com/openai/gym-http-api
# cd gym-http-api
# pip install -r requirements.txt


library(keras)
library(tensorflow)
tf$enable_eager_execution()
library(gym)

remote_base <- ""
client <- create_GymClient(remote_base)
env = gym::env_create(client, "CartPole-v1")
gym::env_list_all(client)

env_reset(client, env)

#action = env_action_space_sample(client, env)
step = env_step(client, env, 1)


env_reset(client, env)
goal_steps = 500
score_requirement = 60
intial_games = 1000



state_size = 4L
action_size = 2L
gamma = 0.95
epsilon = 0.95
epsilon_min = 0.01
epsilon_decay = 0.995
model = keras_model_sequential()
model %>% 
  layer_dense(input_shape = c(4L), units = 20L, activation = "relu") %>% 
  layer_dense(units = 20L, activation = "relu") %>% 
  layer_dense(2L, activation = "linear")
model %>% 
  compile(loss = loss_mean_squared_error, optimizer = optimizer_adamax())


memory = matrix(0, nrow = 8000L, 11L)
counter = 1
remember = function(memory, state, action, reward, next_state, done){
  memory[counter,] = as.numeric(c(state, action, reward, next_state, done))
  counter <<- counter+1
  return(memory)
}

# memory: state 1:4, action 5, reward 6, next_state 7:10, done 11

act = function(state){
  if(runif(1) <= epsilon) return(sample(0:1, 1)) # 
  act_prob = predict(model, matrix(state,nrow = 1L))
  return(which.max(act_prob) -1L)
}

replay = function(batch_size = 32L, memory, counter){
  indices = sample.int(counter, batch_size)
  batch = memory[indices,,drop = FALSE]
  
  for(i in 1:nrow(batch)){
    target = batch[i,6] #reward
    action = batch[i,5] #action
    state = matrix(memory[i, 1:4], nrow = 1L)
    next_state = matrix(memory[i,7:10], nrow =1L)
    if(!batch[i,11]){ # not done
      target = (batch[i,6] + gamma* predict(model, matrix(next_state, nrow = 1L)))[1,1]
    } 
    target_f = predict(model, matrix(state, nrow = 1L))
    target_f[action+1L] = target
    model %>% fit(state, target_f, epochs = 1L, verbose = 0L)
    
    if(epsilon > epsilon_min){
      epsilon <<- epsilon_decay*epsilon
    }
  }
}

done = 0
for(e in 1:100){
  state = unlist(env_reset(client, env))
  
  for(time in 1:500){
    action = act(state)
    response = env_step(client, env, action = action)
    done = as.integer(response$done)
    if(!done) reward = response$reward
    else reward = -10
    
    next_state = unlist(response$observation)
    memory = remember(memory, state, action, reward, next_state, done)
    
    state = next_state
    
    if(done){
      cat("episode", e/500, " score: ", time, " eps: ", epsilon, "\n")
      break()
    } 
    if(counter > 32L) 
      replay(32L, memory, counter-1L)
  }
}
