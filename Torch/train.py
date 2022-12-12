from DQNalign.tool.RL.agent import Agent
from DQNalign.tool.RL.alignment import Pairwise
from DQNalign.tool.RL.Learning import *
import DQNalign.tool.util.ReadSeq as readseq
import DQNalign.tool.Bio.lcs as lcs
from importlib import *
import copy
import time
import os
import json
import torch 

class Dict2Class(object):
    def __init__(self, my_dict):
            for key in my_dict:
                setattr(self, key, my_dict[key])

with open('flags.json') as f: FLAGS = Dict2Class(json.load(f))
with open('param.json') as f: param = Dict2Class(json.load(f)) 
class game_env():
    def __init__(self):
        self.l_seq = [1000, 1000]
        self.win_size = 100
        self.maxI = 10 # maximum indel length
        self.p = [0.1,0.02] # The probability of SNP, indel
        self.reward = [0.5,-0.5,-1] # Alignment score of the match, mismatch, indel
        self.path = "./network/"+FLAGS.model_name+"/"+str(self.win_size)+'_'+str(param.n_step)+"/"+FLAGS.exploration

train_env = game_env()

class model():
    def __init__(self):
        self.param = param
        self.env = Pairwise(train_env,-1,Z=self.param.Z)
        self.LEARNING_RATE = 1 #For win_size 30 to 100 : 1e-3 to 1e-5, win_size 200 : 1e-3 to 1e-5, win_size 500 : 1e-3 to 1e-5, win_Size 1000 : 1e-4 to 1e-5

train_model = model()


seq = readseq.readseqs('lib/HEV.txt')

agent = Agent(FLAGS, True, train_env, train_model)
if FLAGS.resume:
    if len(os.listdir(train_env.path)) == 0 or not os.path.exists(train_env.path):
        raise Exception
    else: 
        agent.load_model(train_env.path, param.iter_to_load)


startdate = time.localtime()

filename = "result/"+FLAGS.model_name+"/training%04d%02d%02d%02d%02d%02d_%d_%d.txt" % (
    startdate.tm_year, startdate.tm_mon, startdate.tm_mday, startdate.tm_hour, startdate.tm_min,
    startdate.tm_sec, train_env.win_size, train_env.maxI)
file = open(filename,"a")
file.write(FLAGS.exploration+"\n")
file.close()

filename2 = "result/"+FLAGS.model_name+"/training2%04d%02d%02d%02d%02d%02d_%d_%d.txt" % (
    startdate.tm_year, startdate.tm_mon, startdate.tm_mday, startdate.tm_hour, startdate.tm_min,
    startdate.tm_sec, train_env.win_size, train_env.maxI)
file = open(filename2,"a")
file.write(FLAGS.exploration+"\n")
file.close()
    
for i in range(train_model.param.num_episodes):
    rT1, rT2, processingtime, j = agent.Global()
    print("Train scenario  :",i, agent.total_steps, rT1, rT2, str(float("{0:.2f}".format(processingtime)))+"s")
    file = open(filename2,"a")
    file.write(str(rT2)+" "+str(processingtime)+"\n")
    file.close()

    # Periodically test the model.
    if i % train_model.param.test_freq == 0 and agent.total_steps > agent.param.pre_train_steps:
        file = open(filename,"a")

        seq1 = seq[0]
        seq2 = seq[1]
        start1,start2,lcslen = lcs.longestSubstring(seq1,seq2)
        if (start1 > 0) and (start2 > 0):
            agent.set(seq1[start1 - 1::-1]+"A", seq2[start2 - 1::-1]+"A")
            rT1, rT2, processingtime1, j = agent.Global()
        else:
            rT1 = 0
            rT2 = 0
            processingtime1 = 0
            j = 0

        rT1o = rT1
        rT2o = rT2
        
        if (start1+lcslen < len(seq1)) and (start2+lcslen < len(seq2)):
            agent.set(seq1[start1+lcslen:]+"A",seq2[start2+lcslen:]+"A")
            rT1, rT2, processingtime2, j = agent.Global()
        else:
            rT1 = 0
            rT2 = 0
            processingtime2 = 0
            j = 0

        print("Test scenario 1 :",i, j, lcslen + rT1o + rT1, lcslen + rT2o + rT2, str(float("{0:.2f}".format(processingtime1+processingtime2)))+"s")
        file.write(str(lcslen + rT2o + rT2)+" "+str(processingtime1+processingtime2)+"\n")

        seq1 = seq[0]
        seq2 = seq[15]
        start1,start2,lcslen = lcs.longestSubstring(seq1,seq2)
        if (start1 > 0) and (start2 > 0):
            agent.set(seq1[start1 - 1::-1]+"A", seq2[start2 - 1::-1]+"A")
            rT1, rT2, processingtime1, j = agent.Global()
        else:
            rT1 = 0
            rT2 = 0
            processingtime1 = 0
            j = 0

        rT1o = rT1
        rT2o = rT2
        
        if (start1+lcslen < len(seq1)) and (start2+lcslen < len(seq2)):
            agent.set(seq1[start1+lcslen:]+"A",seq2[start2+lcslen:]+"A")
            rT1, rT2, processingtime2, j = agent.Global()
        else:
            rT1 = 0
            rT2 = 0
            processingtime2 = 0
            j = 0

        print("Test scenario 2 :",i, j, lcslen + rT1o + rT1, lcslen + rT2o + rT2, str(float("{0:.2f}".format(processingtime1+processingtime2)))+"s")
        file.write(str(lcslen + rT2o + rT2)+" "+str(processingtime1+processingtime2)+"\n")

        agent.reset()
        file.close()
    
    if i % train_model.param.save_freq == 0:
        agent.save_model('network/C51/100_1/e-greedy', i)

