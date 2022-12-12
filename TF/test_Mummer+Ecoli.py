from DQNalign.tool.RL.agent import Agent
from DQNalign.tool.RL.alignment import Pairwise
from DQNalign.tool.RL.Learning import *
import DQNalign.tool.Bio.lcs as lcs
import DQNalign.tool.Bio.conventional as conventional
import DQNalign.tool.util.ReadSeq as readseq
import DQNalign.tool.util.RecordAlign as recordalign
from importlib import *
import copy
import time
import os
import cv2
import tensorflow as tf
import DQNalign.flags as flags



label = os.listdir('../lib/txt')
label = [l[:-4] for l in label]
for l1 in range(len(label)):
    for l2 in range(l1 + 1, len(label)):
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

        seq1 = readseq.readseq('../lib/txt/' + label[l1] + '.txt')
        seq2 = readseq.readseq('../lib/txt/' + label[l2] + '.txt')

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
            rT2o = 0
            jo = 0
            processingtimeo = 0

            #start1,start2,lcslen = lcs.longestSubstring(seq1,seq2)
            if FLAGS.show_align:
                dot_plot = 255*np.ones((len(seq1),len(seq2)))
            if FLAGS.print_align:
                record = recordalign.record_align()

            print("Ecoli test")
            print("raw seq len",len(seq1),len(seq2))
            #print("lcs len 1",start1,lcslen,len(seq1)-start1-lcslen)
            #print("lcs len 2",start2,lcslen,len(seq2)-start2-lcslen)
            past = time.time()

            m = conventional.MUMmer(True,"../lib/fasta/" + label[l1] + ".fasta", "../lib/fasta/" + label[l2] + ".fasta",[label[l1],label[l2]], "./result/C51+Mummer/result%04d%02d%02d%02d%02d%02d_%d_%d_%s_%s_Ecoli" % (
                startdate.tm_year, startdate.tm_mon, startdate.tm_mday, startdate.tm_hour, startdate.tm_min, startdate.tm_sec, train_env.win_size, train_env.maxI, label[l1], label[l2]))
            m.align()
            coords1, coords2, aligns1, aligns2, score = m.export_info()
            now = time.time()
            print("MUMmer result", score, "time", str(now-past)+"s", str(now-start)+"s")

            for i in range(len(coords1)-1):
                processingtime = 0
                if FLAGS.print_align:
                    rT2 = record.record([coords1[i][0],coords1[i][1]+1],[coords2[i][0],coords2[i][1]+1],-1,aligns1[i],aligns2[i])
                else:
                    rT2 = np.sum(np.array(list(aligns1[i]))==np.array(list(aligns2[i])))

                #print(rT2, coords1[i][1]-coords1[i][0])
                rT2o += rT2

                if ((coords1[i+1][0] - coords1[i][1] > 1) and (coords2[i+1][0] - coords2[i][1] > 1)):
                    agent.set(seq1[coords1[i][1]+1:coords1[i+1][0]]+"A", seq2[coords2[i][1]+1:coords2[i+1][0]]+"A")
                    if FLAGS.show_align and FLAGS.print_align:
                        index = np.size(record.xtemp)
                        rT1, rT2, processingtime, j, dot_plot = agent.Global(sess,record)
                        dot_plot[coords1[i][1]+1:coords1[i+1][0],coords2[i][1]+1:coords2[i+1][0]] = dot_plot1
                        record.shift(index,coords1[i][1]+1,coords2[i][1]+1)
                    elif FLAGS.show_align:
                        rT1, rT2, processingtime, j, dot_plot = agent.Global(sess)
                        dot_plot[coords1[i][1]+1:coords1[i+1][0],coords2[i][1]+1:coords2[i+1][0]] = dot_plot1
                    elif FLAGS.print_align:
                        index = np.size(record.xtemp)
                        rT1, rT2, processingtime, j = agent.Global(sess,record)
                        record.shift(index,coords1[i][1]+1,coords2[i][1]+1)
                    else:
                        rT1, rT2, processingtime, j = agent.Global(sess)
                else:
                    rT2 = 0
                    processingtime = 0

                #print(rT2, coords1[i+1][0]-coords1[i][1])
                rT2o += rT2
                processingtimeo += processingtime
                        
                print(i+1, "/", len(coords1), "anchors are finished")
            if score != 0:
                if FLAGS.show_align:
                    # it is not defined yey
                    print("Will added in later...")
                if FLAGS.print_align:
                    rT2 = record.record(coords1[-1],coords2[-1],-1,aligns1[-1],aligns2[-1])
                else:
                    rT2 = np.sum(np.array(list(aligns1[i]))==np.array(list(aligns2[i])))

                rT2o += rT2

                now = time.time()
                #NWresult = np.max(NW.match(alignment.HEVseq[_],alignment.HEVseq[__]))
                print("result", rT2o, "time", str(processingtimeo)+"s", str(now-past)+"s", str(now-start)+"s")
                
                filename = "result/"+FLAGS.model_name+"+MUMmer/result%04d%02d%02d%02d%02d%02d_%d_%d_%s_%s_Ecoli.txt" % (
                    startdate.tm_year, startdate.tm_mon, startdate.tm_mday, startdate.tm_hour, startdate.tm_min,
                    startdate.tm_sec, train_env.win_size, train_env.maxI, label[l1], label[l2])

                file = open(filename,"a")
                file.write(label[l1] + " " + label[l2] + " " + str(score) + " " +  str(rT2o)+" "+str(now-past)+" "+str(now-start)+"\n")
                file.close()

                if FLAGS.show_align:
                    filename = "img/"+FLAGS.model_name+"+MUMmer/result%04d%02d%02d%02d%02d%02d_%d_%d_Ecoli" % (
                        startdate.tm_year, startdate.tm_mon, startdate.tm_mday, startdate.tm_hour, startdate.tm_min,
                        startdate.tm_sec, train_env.win_size, train_env.maxI)
                    cv2.imwrite(filename+".jpg",dot_plot)

                if FLAGS.print_align:
                    filename = "align/"+FLAGS.model_name+"+MUMmer/result%04d%02d%02d%02d%02d%02d_%d_%d_Ecoli.txt" % (
                        startdate.tm_year, startdate.tm_mon, startdate.tm_mday, startdate.tm_hour, startdate.tm_min,
                        startdate.tm_sec, train_env.win_size, train_env.maxI)
                    file = open(filename,"w")

                    file.write("DQNalign Project v1.0\n")
                    file.write("Pairwise alignment algorithm with deep reinforcement learning based heuristic alignment agent\n")
                    file.write("Sequence 1 : " + label[l1] + ", length : "+str(len(seq1))+"\n")
                    file.write("Sequence 2 : " + label[l2] + ", length : "+str(len(seq2))+"\n")
                    file.write("\n")

                    record.print(file)
                    file.close()