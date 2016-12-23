import kagglegym

env = kagglegym.make()
observation = env.reset()

print(len(observation.target))
print(len(observation.train))

n = 0
rewards = []
while True:
   target = observation.target
   target.loc[:, 'y'] = 0.01
   observation, reward, done, info = env.step(target)
   if done:
       break
   rewards.append(reward)
   n = n + 1

print(info)
print(n)
print(rewards[0:15])

# Running
#    python test_gym.py
#
# Should result in:
#
#  968
#  806298
#  {'public_score': -0.42791846067884648}
#  906
# [-0.57244240701870674, -0.69602355761934143, -0.76969605094330085,
# -0.88292556152797097, -0.77409471948462827, -0.62031952966389658,
# -0.39146488525587475, -0.84779897534426341, -0.45051451441360757,
# -0.56453402316570156, -0.89329857536073409, -0.81062686366326098,
# -0.61593843763923251, -0.67321374079183682, -0.82632908490446888]