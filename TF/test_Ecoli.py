from audioop import lin2alaw
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
import tensorflow as tf
import DQNalign.flags as flags
# label = ['AP022811.1.txt', 'CP000133.1.txt', 'Ecoli_1.txt', 'Ecoli_2.txt']
libpath = '../lib/txt/'
label = os.listdir(libpath)
for l1 in range(len(label)):
    for l2 in range(l1+1, len(label)):
        FLAGS = tf.app.flags.FLAGS
        param = import_module('DQNalign.param.'+FLAGS.network_set)

        class game_env():
            def __init__(self):
                self.l_seq = [8000, 8000]
                self.win_size = 100
                self.maxI = 10 # maximum indel length
                self.p = [0.1,0.02] # The probability of SNP, indel
                self.reward = [1,-1,-1] # Alignment score of the match, mismatch, indel

                self.path = "./network/"+FLAGS.model_name+"/"+str(self.win_size)+'_'+str(param.n_step)

        train_env = game_env()

        class model():
            def __init__(self):
                self.param = param
                self.env = Pairwise(train_env,-1,Z=self.param.Z)
                self.LEARNING_RATE = 0.0000001

                tf.reset_default_graph()
                
                """ Define Deep reinforcement learning network """
                if FLAGS.model_name == "DQN":
                    self.mainQN = Qnetwork(self.param.h_size,self.env,self.LEARNING_RATE,self.param.n_step)
                    self.targetQN = Qnetwork(self.param.h_size,self.env,self.LEARNING_RATE,self.param.n_step)
                    self.trainables = tf.trainable_variables()
                    self.targetOps = updateTargetGraph(self.trainables, self.param.tau)
                elif FLAGS.model_name == "SSD":
                    self.mainQN = SSDnetwork(self.param.h_size,self.env,"main",self.LEARNING_RATE,self.param.n_step)
                    self.targetQN = SSDnetwork(self.param.h_size,self.env,"target",self.LEARNING_RATE,self.param.n_step)
                    self.trainables = tf.trainable_variables()
                    self.targetOps = updateTargetGraph(self.trainables, self.param.tau)
                elif FLAGS.model_name == "C51":
                    self.mainQN = C51(self.param.h_size, self.env, "main", self.LEARNING_RATE, self.param.n_step, 51)
                    self.targetQN = C51(self.param.h_size, self.env, "target", self.LEARNING_RATE, self.param.n_step, 51)
                    self.trainables = tf.trainable_variables()
                    self.targetOps = updateTargetGraph(self.trainables, self.param.tau)

                self.init = tf.global_variables_initializer()
                self.saver = tf.train.Saver()

        train_model = model()
        init = train_model.init
        saver = train_model.saver


        seq1 = readseq.readseq(libpath+label[l1])
        seq2 = readseq.readseq(libpath+label[l2])

        if not os.path.exists(train_env.path):
            os.makedirs(train_env.path)

        if np.size(os.listdir(train_env.path)) > 0:
            resume = FLAGS.resume
        else:
            resume = False

        """ Main test step """
        with tf.Session() as sess:
            if FLAGS.use_GPU:
                sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
            else:
                sess = tf.Session(config=tf.ConfigProto(device_count={'CPU': 0}))

            sess.run(init)
            agent = Agent(FLAGS, False, train_env, train_model)

            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(train_env.path + '/e-greedy')
            saver.restore(sess, ckpt.model_checkpoint_path)
            
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
                rT1, rT2, processingtime, j, dot_plot = agent.Global(sess,record)
            elif FLAGS.show_align:
                rT1, rT2, processingtime, j, dot_plot = agent.Global(sess)
            elif FLAGS.print_align:
                rT1, rT2, processingtime, j = agent.Global(sess,record)
            else:
                rT1, rT2, processingtime, j = agent.Global(sess)

            now = time.time()
            #NWresult = np.max(NW.match(alignment.HEVseq[_],alignment.HEVseq[__]))
            print("result", rT2, "time", str(now-past)+"s", str(now-start)+"s")
            
            filename = "result/"+FLAGS.model_name+"/result%04d%02d%02d%02d%02d%02d_%d_%d_Ecoli.txt" % (
                startdate.tm_year, startdate.tm_mon, startdate.tm_mday, startdate.tm_hour, startdate.tm_min,
                startdate.tm_sec, train_env.win_size, train_env.maxI)

            file = open(filename,"a")
            file.write(label[l1] + " " + label[l2] + " " + str(rT2)+" "+str(now-past)+" "+str(now-start)+"\n")
            file.close()

            if FLAGS.show_align:
                filename = "img/"+FLAGS.model_name+"/result%04d%02d%02d%02d%02d%02d_%d_%d_Ecoli" % (
                    startdate.tm_year, startdate.tm_mon, startdate.tm_mday, startdate.tm_hour, startdate.tm_min,
                    startdate.tm_sec, train_env.win_size, train_env.maxI)
                cv2.imwrite(filename+".jpg",dot_plot)

            if FLAGS.print_align:
                filename = "align/"+FLAGS.model_name+"/"+label[l1]+"_"+label[l2]+'.txt'
                file = open(filename,"w")

                file.write("DQNalign Project v1.0\n")
                file.write("Pairwise alignment algorithm with deep reinforcement learning based heuristic alignment agent\n")
                file.write("Sequence 1 : " + label[l1] + ", length : "+str(len(seq1))+"\n")
                file.write("Sequence 2 : " + label[l2] + ", length : "+str(len(seq2))+"\n")
                file.write("\n")

                record.print(file)

                file.close()
