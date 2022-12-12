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
import torch
import json
import sys

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

""" Main test step """

agent = Agent(FLAGS, False, train_env, train_model)
if FLAGS.resume:

    try:
        iteration = int(sys.argv[1])
    except Exception as e:
        print(e)
        iteration = param.iter_to_load
    
    try:
        os.mkdir(f'align/{FLAGS.model_name}/epoch_{iteration}')
    except Exception as e:
        print(e)
    
    agent.load_model('network/C51/100_1/e-greedy/', iteration)
print('Loading Model...')
print('######################')

start = time.time()
startdate = time.localtime()

for _ in range(47):
    #print(_)
    for __ in range(_+1,47):
        seq1 = seq[_]
        seq2 = seq[__]
        start1,start2,lcslen = lcs.longestSubstring(seq1,seq2)

        if FLAGS.show_align:
            dot_plot = 255*np.ones((len(seq1),len(seq2)))
            for i in range(lcslen):
                dot_plot[start1+i,start2+i]=0
        if FLAGS.print_align:
            record = recordalign.record_align()
            record.set_test_type('SSD')

        print("test",_,__)
        print("raw seq len",len(seq1),len(seq2))
        print("lcs len 1",start1,lcslen,len(seq1)-start1-lcslen)
        print("lcs len 2",start2,lcslen,len(seq2)-start2-lcslen)
        past = time.time()

        if (start1 > 0) and (start2 > 0):
            agent.set(seq1[start1 - 1::-1]+"A", seq2[start2 - 1::-1]+"A")
            if FLAGS.show_align and FLAGS.print_align:
                rT1, rT2, processingtime, j, dot_plot1 = agent.Global(record)
                dot_plot[:start1,:start2] = dot_plot1[::-1,::-1]
                record.reverse(start1-1,start2-1)
            elif FLAGS.show_align:
                rT1, rT2, processingtime, j, dot_plot1 = agent.Global()
                dot_plot[:start1,:start2] = dot_plot1[::-1,::-1]
            elif FLAGS.print_align:
                rT1, rT2, processingtime, j = agent.Global(record)
                record.reverse(start1-1,start2-1)
            else:
                rT1, rT2, processingtime, j = agent.Global()
        else:
            rT1 = 0
            rT2 = 0
            processingtime = 0
            j = 0

        rT2o = rT2
        if FLAGS.print_align:
            record.record([start1,start1+lcslen],[start2,start2+lcslen],-1,seq1[start1:start1+lcslen],seq2[start2:start2+lcslen])
        
        if (start1+lcslen < len(seq1)) and (start2+lcslen < len(seq2)):
            agent.set(seq1[start1+lcslen:]+"A",seq2[start2+lcslen:]+"A")
            if FLAGS.show_align and FLAGS.print_align:
                index = np.size(record.xtemp)
                rT1, rT2, processingtime, j, dot_plot2 = agent.Global(record)
                record.shift(index,start1+lcslen,start2+lcslen)
                dot_plot[start1+lcslen:,start2+lcslen:] = dot_plot2
            elif FLAGS.show_align:
                rT1, rT2, processingtime, j, dot_plot2 = agent.Global()
                dot_plot[start1+lcslen:,start2+lcslen:] = dot_plot2
            elif FLAGS.print_align:
                index = np.size(record.xtemp)
                rT1, rT2, processingtime, j = agent.Global(record)
                record.shift(index,start1+lcslen,start2+lcslen)
            else:
                rT1, rT2, processingtime, j = agent.Global()
        else:
            rT1 = 0
            rT2 = 0
            processingtime = 0
            j = 0
        now = time.time()
        #NWresult = np.max(NW.match(alignment.HEVseq[_],alignment.HEVseq[__]))
        print("result", lcslen + rT2o + rT2, "rawdata", rT2o, lcslen, rT2, "time", str(now-past)+"s", str(now-start)+"s")

            
        filename = "result/"+FLAGS.model_name+"/result%04d%02d%02d%02d%02d%02d_%d_%d.txt" % (
            startdate.tm_year, startdate.tm_mon, startdate.tm_mday, startdate.tm_hour, startdate.tm_min,
            startdate.tm_sec, train_env.win_size, train_env.maxI)

        file = open(filename,"a")
        file.write(str(lcslen + rT2o + rT2)+" "+str(now-past)+" "+str(now-start)+"\n")
        file.close()

        if FLAGS.show_align:
            filename = "img/"+FLAGS.model_name+"/result%04d%02d%02d%02d%02d%02d_%d_%d" % (
                startdate.tm_year, startdate.tm_mon, startdate.tm_mday, startdate.tm_hour, startdate.tm_min,
                startdate.tm_sec, train_env.win_size, train_env.maxI)
            cv2.imwrite(filename+"_"+str(_)+"_"+str(__)+".jpg",dot_plot)

        if FLAGS.print_align:

            filename = f'align/{FLAGS.model_name}/epoch_{iteration}/result_{_}_{__}.txt' 
            # filename = "align/"+FLAGS.model_name+"/epoch_%d/result%04d%02d%02d%02d%02d%02d_%d_%d.txt" % (train_env.win_size, train_env.maxI,
            #     startdate.tm_year, startdate.tm_mon, startdate.tm_mday, startdate.tm_hour, startdate.tm_min,
            #     startdate.tm_sec, _, __)
            file = open(filename,"w")
            
            file.write("DQNalign Project v1.0\n")
            file.write("Pairwise alignment algorithm with deep reinforcement learning based heuristic alignment agent\n")
            file.write("Sequence 1 : HEV_"+str(_)+", length : "+str(len(seq1))+"\n")
            file.write("Sequence 2 : HEV_"+str(__)+", length : "+str(len(seq2))+"\n")
            file.write("\n")

            record.print(file)
            to_save = {}
            if FLAGS.test_identity:
                identity_matches = 0
                to_save['Seq1_name'] = "HEV_"+str(_)
                to_save['Seq2_name'] = "HEV_"+str(__)
                to_save['Seq1_len'] = str(len(seq1))
                to_save['Seq2_len'] = str(len(seq2))
                to_save['Seq1'] = str(seq1)
                to_save['Seq2'] = str(seq2)
                to_save['Seq1_align'] = str(record.Qtemp)
                to_save['Seq2_align'] = str(record.Stemp)
                to_save['Exact_matches'] =  str(lcslen + rT2o + rT2)   
                to_save['record_exact_matches'] = str(record.stemp)
                to_save['record_gaps'] = str(record.gtemp)
                with open('identity/SSD/identity_HEV.text', 'a') as f: json.dump(to_save, f)
                    
            file.close()

