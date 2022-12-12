from collections import deque
import numpy as np
# import tensorflow as tf
import os
from importlib import *
from DQNalign.tool.RL.alignment import Pairwise
from DQNalign.tool.RL.Learning import *
import DQNalign.tool.util.function as function
import time
import torch
import copy
import json
#from tool.RL.Learning import per_experience_buffer

USE_PER = True
PER_BATCH_SIZE = 32
class Dict2Class(object):
    def __init__(self, my_dict):
            for key in my_dict:
                setattr(self, key, my_dict[key])

class Agent():
    def __init__(self, FLAGS, istrain, game_env, model, seq1 = [], seq2 = [], ismeta = False):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        """ Get parameters from files """
        self.FLAGS = FLAGS
        self.istrain = istrain

        with open('param.json') as f: self.param = Dict2Class(json.load(f)) 

        """ Exploration strategy """
        if self.istrain:
            self.l_seq = game_env.l_seq
            self.e = self.param.startE
            self.stepDrop = (self.param.startE - self.param.endE) / self.param.annealing_steps

        """ Define sequence alignment environment """
        if self.istrain:
            self.env = Pairwise(game_env,0,Z=self.param.Z)
        else:
            if len(seq1)+len(seq2) > 0:
                self.env = Pairwise(game_env,1,seq1,seq2,Z=self.param.Z)
            else:
                self.env = Pairwise(game_env,0,Z=self.param.Z)

        self.num_atoms = 51
        self.num_actions = 3
        self.vmax = 10 
        self.vmin = -10
        self.supports = torch.linspace(self.vmin, self.vmax, self.num_atoms).view(1,1,self.num_atoms).numpy()
        self.tsupports = torch.linspace(self.vmin, self.vmax, self.num_atoms).view(1,1,self.num_atoms).to(self.device)
        self.delta_z = (self.vmax - self.vmin)/(self.num_atoms - 1.0)
        self.z = np.array([self.vmin + i*self.delta_z for i in range(self.num_atoms)])

        if FLAGS.model_name == 'C51':
            self.mainQN : C51 = C51(self.param.h_size, self.env, 'main', 51, 3)
            self.targetQN : C51 = C51(self.param.h_size, self.env, 'target', 51, 3)
            self.mainQN = self.mainQN.to(self.device)
            self.targetQN = self.targetQN.to(self.device)
            self.optimizer = torch.optim.Adam(params=self.mainQN.parameters(), lr=1e-4)
        
        """ Initialize the variables """
        self.total_steps = 0
        self.start = time.time()
        if USE_PER:
            self.myBuffer = per_experience_buffer()
        else:
            self.myBuffer = experience_buffer()
    
    def save_model(self, path: str, iteration: int) -> None:
        torch.save(self.mainQN.state_dict(), f'{path}/model_{iteration}.dump')
        torch.save(self.optimizer.state_dict(), f'{path}/optimizer_{iteration}.dump')
    
    def load_model(self, path: str, iteration: int) -> None:
        self.mainQN.load_state_dict(torch.load(f'{path}/model_{iteration}.dump'))
        self.targetQN.load_state_dict(self.mainQN.state_dict())
        self.optimizer.load_state_dict(torch.load(f'{path}/optimizer_{iteration}.dump'))
    
    def update_lr(self):
        for g in self.optimizer.param_groups:
            g['lr'] = g['lr']*0.9
    
    def get_action(self, state):
        with torch.no_grad():
            x = self.mainQN(state).detach().cpu().numpy()
            x = np.sum(np.multiply(np.vstack(x), self.z), axis=1)
            return np.argmax(x)

    def reset(self):
        """ Define sequence alignment environment """
        self.istrain = True
        self.env.sizeS1 = self.l_seq[0]
        self.env.sizeS2 = self.l_seq[1]

    def set(self, seq1 = [], seq2 = []):
        """ Define sequence alignment environment """
        self.istrain = False
        self.env.test(seq1,seq2)

    def train(self):
        if USE_PER:
            tree_idx, trainBatch = self.myBuffer.sample(PER_BATCH_SIZE)
            trainBatch = np.array(trainBatch)
        else:
            trainBatch = self.myBuffer.sample(self.param.batch_size)  # Select the batch from the experience buffer
        #print(np.shape(np.vstack(trainBatch[:, 3])))
        
        if self.FLAGS.model_name == 'C51':
            states = np.vstack(trainBatch[:, 0])
            actions = np.array(trainBatch[:, 1])
            rewards = np.array(trainBatch[:, 2])
            next_states = np.vstack(trainBatch[:, 3])
            done = np.array(trainBatch[:, 4])
            m_prob = np.array([np.zeros((self.param.batch_size, self.num_atoms)) for i in range(3)])
            optimal_action_idx = list()

        
            Q1 = self.mainQN(torch.from_numpy(next_states).to(self.device)).cpu().detach().numpy()
            # with torch.no_grad():
            Q2 = self.targetQN(torch.from_numpy(next_states).to(self.device)).cpu().detach().numpy()
            
            q = np.sum(np.multiply(np.vstack(Q1), self.z), axis=1)                
            # print(q.shape)

            q = q.reshape((self.param.batch_size, 3), order='F')
            optimal_actions_idx = np.argmax(q, axis=1)
            for i in range(self.param.batch_size):
                if done[i]:
                    Tz = min(self.vmax, max(self.vmin, rewards[i]))
                    bj = (Tz - self.vmin)/self.delta_z
                    m_l, m_u = np.floor(bj), np.ceil(bj)
                    m_prob[actions[i]][i][int(m_l)] += (m_u - bj)
                    m_prob[actions[i]][i][int(m_u)] += (bj - m_l)
                else: 
                    for j in range(self.num_atoms):
                        Tz = min(self.vmax, max(self.vmin, rewards[i] + (0.99**self.param.n_step) * self.z[j]))
                        bj = (Tz - self.vmin)/self.delta_z
                        m_l, m_u = np.floor(bj), np.ceil(bj)
                        m_prob[actions[i]][i][int(m_l)] += Q2[optimal_actions_idx[i]][i][j]*(m_u -bj)
                        m_prob[actions[i]][i][int(m_u)] += Q2[optimal_actions_idx[i]][i][j]*(bj - m_l)
            if USE_PER:
                loss = self.compute_loss_c51(torch.from_numpy(states).to(self.device), torch.from_numpy(m_prob).to(self.device), tree_idx).cpu().detach().numpy()
            else:
                loss = self.compute_loss_c51(torch.from_numpy(states).to(self.device), torch.from_numpy(m_prob).to(self.device)).cpu().detach().numpy().numpy()
            
            self.soft_target_update(0.001)
        
    def soft_target_update(self, tau : float):
        for target_param, local_param in zip(self.targetQN.parameters(), self.mainQN.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)            # for target_param, local_param in zip(self.targetQN.parameters(), self.mainQN.parameters()):
            #     target_param.data.copy_(tau*local_param.data + (1.0 - tau)*target_param.data)

    def per_error(self, y_true : torch.Tensor, y_pred : torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.nn.functional.kl_div(y_pred, y_true, reduce=False)
    def compute_loss_c51(self, states : torch.Tensor, probs: torch.Tensor, tree_idx=None) -> torch.Tensor:
        pred : torch.Tensor = self.mainQN(states)
        loss = (pred.log()*probs).sum(0)
        if USE_PER:
            to_add = loss.detach().squeeze().negative().cpu().numpy()
            to_add = np.mean(to_add, axis=1)
            self.myBuffer.batch_update(tree_idx, to_add)
        
        loss = loss.mean().negative()
        self.optimizer.zero_grad()
        loss.backward()
        self.mainQN.reset_noise()
        self.targetQN.reset_noise()
        self.optimizer.step()
        return loss

    def skip(self):
        a = []

        seq1end = min(self.env.x+self.env.win_size-1,self.env.sizeS1-1)
        seq2end = min(self.env.y+self.env.win_size-1,self.env.sizeS2-1)
        minend = min(seq1end-self.env.x,seq2end-self.env.y)
        diff = np.where(self.env.seq1[self.env.x:self.env.x + minend + 1] != self.env.seq2[self.env.y:self.env.y + minend + 1])
        if np.size(diff) > 0:
            a = [0] * np.min(diff)
        else:
            a = [0] * minend

        return a

    def reverseskip(self):
        a = []

        seq1end = max(self.env.x-self.env.win_size+1,0)
        seq2end = max(self.env.y-self.env.win_size+1,0)
        minend = min(self.env.x-seq1end,self.env.y-seq2end)
        diff = np.where(self.env.seq1[self.env.x-minend:self.env.x + 1][::-1] != self.env.seq2[self.env.y-minend:self.env.y + 1][::-1])
        if np.size(diff) > 0:
            a = [0] * np.max(diff)
        else:
            a = [0] * minend

        return a

    def skipRC(self):
        a = []

        seq1end = min(self.env.x+self.env.win_size-1,self.env.sizeS1-1)
        seq2end = max(self.env.y-self.env.win_size+1,0)
        minend = min(seq1end-self.env.x,self.env.y-seq2end)
        diff = np.where(self.env.seq1[self.env.x:self.env.x + minend + 1] != self.env.rev2[self.env.y-minend:self.env.y + 1][::-1])
        if np.size(diff) > 0:
            a = [0] * np.min(diff)
        else:
            a = [0] * minend

        return a

    def reverseskipRC(self):
        a = []

        seq1end = max(self.env.x-self.env.win_size+1,0)
        seq2end = min(self.env.y+self.env.win_size-1,self.env.sizeS2-1)
        minend = min(self.env.x-seq1end,seq2end-self.env.y)
        diff = np.where(self.env.seq1[self.env.x-minend:self.env.x + 1][::-1] != self.env.rev2[self.env.y:self.env.y + minend + 1])
        if np.size(diff) > 0:
            a = [0] * np.max(diff)
        else:
            a = [0] * minend

        return a

    def add_to_replay(self, step_buffer, episodeBuffer=None):
        with torch.no_grad():
            s, a, r, obs, d = step_buffer[-1]

            if len(step_buffer) < self.param.n_step:
                return 
            
            R = sum([step_buffer[i][2]*(0.99**i) for i in range(self.param.n_step)])
            state, action, _, _, _ = step_buffer.pop()
            if USE_PER: self.myBuffer.add(np.reshape(np.array([state, action, R, obs, d]), [1, 5]))
            else: episodeBuffer.add(np.reshape(np.array([state, action, R, obs, d]), [1, 5]))  # Save the result into episode buffer

    def Global(self, record=0):
        # Newly define experience buffer for new episode
        step_buffer = list()
        past = time.time()
        self.mainQN.set_training(self.istrain)
        if self.FLAGS.show_align:
            dot_plot = 255*np.ones((self.env.sizeS1,self.env.sizeS2))
        if self.FLAGS.print_align:
            Nucleotide = ["N","A","C","G","T"]
        if self.istrain:
            if USE_PER:
                episodeBuffer = []
            else:
                episodeBuffer = experience_buffer()
            # Environment reset for each episode
            s1 = self.env.reset() # Rendered image of the alignment environment
            s1 = processState(s1) # Resize to 1-dimensional vector
        else:
            s = processState(self.env.renderEnv())
        d = False # The state of the game (End or Not)
        rT1 = 0 # Total reward
        rT2 = 0 # Total match
        j = 0

        while j < self.env.sizeS1+self.env.sizeS2:  # Training step is proceeded until the maximum episode length
            # print(self.istrain)
            #print(self.env.x, self.env.y)
            if self.FLAGS.display_process:
                if j % 1000 == 0:
                    now = time.time()

            # Exploration step
            if self.env.seq1[self.env.x]==self.env.seq2[self.env.y]:
                a = self.skip()[0]
            elif self.istrain:
                s1 = processState(self.env.renderEnv())
                a = self.get_action(torch.from_numpy(np.array(s1)).to(self.device))
            else:
                s1 = processState(self.env.renderEnv())
                a = self.get_action(torch.from_numpy(np.array(s1)).to(self.device))


            # Update the DQN network
            if self.istrain:
                s = s1
                s1, r, d = self.env.step(a)
                j += 1
                s1 = processState(s1)
                step_buffer.append([s, a, r, s1, d])
                self.total_steps += 1
                rT1 += r
                rT2 += (r>0)
                if USE_PER: self.add_to_replay(step_buffer) 
                else: self.add_to_replay(step_buffer, episodeBuffer=episodeBuffer) 
                if self.total_steps > self.param.pre_train_steps:
                    if self.e > self.param.endE:
                        self.e -= self.stepDrop

                    if self.total_steps % (self.param.update_freq) == 0:
                        self.train()
            else:
                for _ in range(np.size(a)):
                    if self.FLAGS.show_align:
                        dot_plot[self.env.x][self.env.y] = 0
                    if self.FLAGS.print_align:
                        record.record(self.env.x, self.env.y, a, Nucleotide[self.env.seq1[self.env.x]+1], Nucleotide[self.env.seq2[self.env.y]+1])

                    r, d = self.env.teststep(a)
                    j += 1
                    rT1 += r
                    rT2 += (r>0)
                    if d == True:
                        break

            if d == True:
                break

            if self.FLAGS.display_process:
                if j % 1000 == 1000-1:
                    print("Align step is processed :",j+1,"with",time.time()-now)
    
        if self.istrain:
            if not USE_PER:
                self.myBuffer.add(episodeBuffer.buffer)
            # if USE_PER:
            #     for experience in episodeBuffer:
            #         self.myBuffer.add(experience)
            # else:
            #     self.myBuffer.add(episodeBuffer.buffer)

        now = time.time()
        if self.FLAGS.show_align and self.FLAGS.print_align:
            return rT1, rT2, now - past, j, dot_plot
        elif self.FLAGS.show_align:
            return rT1, rT2, now - past, j, dot_plot
        elif self.FLAGS.print_align:
            return rT1, rT2, now - past, j
        return rT1, rT2, now - past, j

