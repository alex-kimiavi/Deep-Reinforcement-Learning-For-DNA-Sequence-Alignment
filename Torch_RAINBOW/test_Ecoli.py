from DQNalign.tool.RL.agent import Agent
from DQNalign.tool.RL.alignment import Pairwise
from DQNalign.tool.RL.Learning import *
import DQNalign.tool.Bio.lcs as lcs
import DQNalign.tool.util.ReadSeq as readseq
import DQNalign.tool.util.RecordAlign as recordalign
from importlib import *
import copy
import time
import os
import cv2
import json

class Dict2Class(object):
    def __init__(self, my_dict):
            for key in my_dict:
                setattr(self, key, my_dict[key])

with open('flags.json') as f: FLAGS = Dict2Class(json.load(f))
with open('param.json') as f: param = Dict2Class(json.load(f)) 

libpath = '../lib/txt/'
label = os.listdir(libpath)
print(label)
for l1 in range(len(label)):
    for l2 in range(l1+1, len(label)):
        class game_env():
            def __init__(self):
                self.l_seq = [1000, 1000]
                self.win_size = 100
                self.maxI = 10 # maximum indel length
                self.p = [0.1,0.02] # The probability of SNP, indel
                self.reward = [1,-1,-1] # Alignment score of the match, mismatch, indel

                self.path = "./network/"+FLAGS.model_name+"/"+str(self.win_size)+'_'+str(param.n_step)+'/e-greedy/'

        train_env = game_env()

        class model():
            def __init__(self):
                self.param = param
                self.env = Pairwise(train_env,-1,Z=self.param.Z)
                self.LEARNING_RATE = 0.0000001

        train_model = model()


        seq1 = readseq.readseq(libpath+label[l1])
        seq2 = readseq.readseq(libpath+label[l2])

        if not os.path.exists(train_env.path):
            os.makedirs(train_env.path)

        if np.size(os.listdir(train_env.path)) > 0:
            resume = FLAGS.resume
        else:
            resume = False
        print("####################################")
        print(resume)
        print("###############################")
        """ Main test step """


        agent = Agent(FLAGS, False, train_env, train_model)

        if FLAGS.resume:
            if len(os.listdir(train_env.path)) == 0 or not os.path.exists(train_env.path):
                raise Exception
            else: 
                agent.load_model(train_env.path, param.iter_to_load)
        start = time.time()
        startdate = time.localtime()

        #start1,start2,lcslen = lcs.longestSubstring(seq1,seq2)
        if FLAGS.print_align:
            record = recordalign.record_align()

        print("Ecoli test")
        print("raw seq len",len(seq1),len(seq2))
        #print("lcs len 1",start1,lcslen,len(seq1)-start1-lcslen)
        #print("lcs len 2",start2,lcslen,len(seq2)-start2-lcslen)
        past = time.time()

        agent.set(seq1, seq2)
        if FLAGS.show_align and FLAGS.print_align:
            rT1, rT2, processingtime, j, dot_plot = agent.Global(record)
        elif FLAGS.show_align:
            rT1, rT2, processingtime, j, dot_plot = agent.Global()
        elif FLAGS.print_align:
            rT1, rT2, processingtime, j = agent.Global(record)
        else:
            rT1, rT2, processingtime, j = agent.Global()

        now = time.time()
        print("result", rT2, "time", str(now-past)+"s", str(now-start)+"s")
        
        filename = f"result/{FLAGS.model_name}/{label[l1][:-4]}_{label[l2][:-4]}_Ecoli.txt" 

        file = open(filename,"a")
        file.write(label[l1][:-4] + " " + label[l2][:-4] + " " + str(rT2)+" "+str(now-past)+" "+str(now-start)+"\n")
        file.close()

        if FLAGS.show_align:
            filename = "img/"+FLAGS.model_name+"/result%04d%02d%02d%02d%02d%02d_%d_%d_Ecoli" % (
                startdate.tm_year, startdate.tm_mon, startdate.tm_mday, startdate.tm_hour, startdate.tm_min,
                startdate.tm_sec, train_env.win_size, train_env.maxI)
            cv2.imwrite(filename+".jpg",dot_plot)

        if FLAGS.print_align:
            print(record.stemp)
            filename = "align/"+FLAGS.model_name+"/C51_PER_DUEL_100_"+label[l1]+"_"+label[l2]+'.txt'
            file = open(filename,"w")

            file.write("DQNalign Project v1.0\n")
            file.write("Pairwise alignment algorithm with deep reinforcement learning based heuristic alignment agent\n")
            file.write("Sequence 1 : " + label[l1] + ", length : "+str(len(seq1))+"\n")
            file.write("Sequence 2 : " + label[l2] + ", length : "+str(len(seq2))+"\n")
            file.write("\n")

            record.print(file)

            file.close()
