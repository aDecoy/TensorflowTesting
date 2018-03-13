from Evolutionary_Strat.acrobot_es.acrobat_agent import *
import numpy


agent = Agent()



# the pre-trained weights are saved into 'weights.pkl' which you can use.
# agent.load()


for i in range(10):
    print('Age {}'.format(i))
    agent.train(300)
    agent.save('acrobat.pkl')

agent.play(20)

