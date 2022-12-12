import numpy as np
import scipy.misc
import DQNalign.tool.util.function as function
import matplotlib.pyplot as plt
import time

np.set_printoptions(threshold=np.inf)

class Pairwise():
    def __init__(self, env, str, seq1=[], seq2=[], Z=3):
        # reward must be defined as the score of [match, mismatch, indel]
        self.reward = env.reward
        self.sizeS1 = env.l_seq[0]
        self.sizeS2 = env.l_seq[1]
        self.win_size = env.win_size
        #self.freq_size = env.freq_size
        #self.K = env.K
        self.p = env.p
        self.maxI = env.maxI
        self.actions = 3
        self.Z = Z
        if str==0:
            a = self.reset()
        elif str==1:
            a = self.test(seq1,seq2)
        elif str==2:
            a = self.reset(2)
        elif str==3:
            a = self.reset(3)
        #plt.imshow(a, interpolation="nearest", cmap="rainbow")
        #plt.show()

    def reset(self, mode=1):
        self.x = 0
        self.y = 0
        self.seq1 = np.random.randint(4, size=self.sizeS1)
        self.seq2 = np.mod(self.seq1 + (np.random.rand(self.sizeS1) < self.p[0]) * np.random.randint(4, size=self.sizeS1), 4)
        count1 = 0
        count2 = 0
        for kk in range(self.sizeS1):
            if np.random.rand() < self.p[1]:
                indel = function.zipfian(1.6, self.maxI)
                ranval = np.random.rand()
                if ranval < 1 / 2:
                    temp1 = self.seq1[0:kk + count1]
                    temp4 = self.seq1[kk + count1:]
                    self.seq1 = np.append(np.append(temp1, np.random.randint(4, size=indel)), temp4)
                    count1 = count1 + indel
                else:
                    temp2 = self.seq2[0:kk + count2]
                    temp5 = self.seq2[kk + count2:]
                    self.seq2 = np.append(np.append(temp2, np.random.randint(4, size=indel)), temp5)
                    count2 = count2 + indel

        if np.size(self.seq1) >= self.sizeS1:
            self.seq1 = self.seq1[0:self.sizeS1]
        else:
            tempseq = np.random.randint(4, size=self.sizeS1-np.size(self.seq1))
            self.seq1 = self.seq1.append(tempseq)

        if np.size(self.seq2) >= self.sizeS2:
            self.seq2 = self.seq2[0:self.sizeS2]
        else:
            tempseq = np.random.randint(4, size=self.sizeS2-np.size(self.seq2))
            self.seq2 = self.seq2.append(tempseq)

        if mode == 1:
            self.state = self.renderEnv()
        elif mode == 2:
            self.state = self.renderDiff()
        elif mode == 3:
            self.state = self.renderFFT()

        return self.state

    def test(self, seq1, seq2):
        if np.size(seq1)>0 and np.size(seq2)>0:
            if (seq1[0] == 0) or (seq1[0] == 1) or (seq1[0] == 2) or (seq1[0] == 3) or (seq1[0] == -1) :
                seqTemp1 = seq1
                seqTemp2 = seq2
                self.seq1 = seqTemp1.astype(int)
                self.seq2 = seqTemp2.astype(int)
                self.rev2 = (3-self.seq2[::-1]) - 5 * (self.seq2[::-1]==-1)
            else:
                seqTemp1 = np.zeros(len(seq1), dtype=int)
                for _ in range(len(seqTemp1)):
                    seqTemp1[_] = (seq1[_] == 'A') + 2 * (seq1[_] == 'C') + 3 * (seq1[_] == 'G') + 4 * (
                            seq1[_] == 'T') - 1
                seqTemp2 = np.zeros(len(seq2), dtype=int)
                for _ in range(len(seqTemp2)):
                    seqTemp2[_] = (seq2[_] == 'A') + 2 * (seq2[_] == 'C') + 3 * (seq2[_] == 'G') + 4 * (
                            seq2[_] == 'T') - 1
                self.seq1 = seqTemp1.astype(int)
                self.seq2 = seqTemp2.astype(int)
                self.rev2 = (3-self.seq2[::-1]) - 5 * (self.seq2[::-1]==-1)
        else:
            self.seq1 = seq1
            self.seq2 = seq2
            self.rev2 = (3-self.seq2[::-1]) - 5 * (self.seq2[::-1]==-1)
        self.x = 0
        self.y = 0
        self.sizeS1 = np.size(self.seq1)
        self.sizeS2 = np.size(self.seq2)

        self.state = self.renderEnv()
        #print("alignlen", self.sizeS1, self.sizeS2)

        return self.state

    def moveChar(self, action):
        # 0 - Match, 1 - Seq1 Insertion, 2 - Seq2 Insertion, 10 - Reverse Match, 11 - Seq1 Reverse Insertion, 12 - Seq2 Reverse Insertion
        if action == 0:
            if self.seq1[self.x] == self.seq2[self.y]:
                reward = self.reward[0]
            else:
                reward = self.reward[1]
            self.x += 1
            self.y += 1

        if action == 1:
            reward = self.reward[2]
            self.x += 1

        if action == 2:
            reward = self.reward[2]
            self.y += 1

        if action == 10:
            if self.seq1[self.x] == self.seq2[self.y]:
                reward = self.reward[0]
            else:
                reward = self.reward[1]
            self.x -= 1
            self.y -= 1

        if action == 11:
            reward = self.reward[2]
            self.x -= 1

        if action == 12:
            reward = self.reward[2]
            self.y -= 1

        if ((self.x >= self.sizeS1-1) and (action < 10)) or ((self.x <= 0) and (action >=10)):
            done = True
        elif ((self.y >= self.sizeS2-1) and (action < 10)) or ((self.y <= 0) and (action >=10)):
            done = True
        else:
            done = False

        if self.seq1[self.x] < 0 and self.seq2[self.y] < 0:
            done = 0
        #print(self.x)
        #print(self.y)
        #print(done)
        return reward, done

    def moveCharRC(self, action):
        # 0 - Match, 1 - Seq1 Insertion, 2 - Seq2 Insertion, 10 - Reverse Match, 11 - Seq1 Reverse Insertion, 12 - Seq2 Reverse Insertion
        if action == 0:
            if self.seq1[self.x] == self.rev2[self.y]:
                reward = self.reward[0]
            else:
                reward = self.reward[1]
            self.x += 1
            self.y -= 1

        if action == 1:
            reward = self.reward[2]
            self.x += 1

        if action == 2:
            reward = self.reward[2]
            self.y -= 1

        if action == 10:
            if self.seq1[self.x] == self.rev2[self.y]:
                reward = self.reward[0]
            else:
                reward = self.reward[1]
            self.x -= 1
            self.y += 1

        if action == 11:
            reward = self.reward[2]
            self.x -= 1

        if action == 12:
            reward = self.reward[2]
            self.y += 1

        if ((self.x >= self.sizeS1-1) and (action < 10)) or ((self.x <= 0) and (action >=10)):
            done = True
        elif ((self.y >= self.sizeS2-1) and (action < 10)) or ((self.y <= 0) and (action >=10)):
            done = True
        else:
            done = False

        if self.seq1[self.x] < 0 and self.seq2[self.y] < 0:
            done = 0

        #print(self.x)
        #print(self.y)
        #print(done)
        return reward, done

    def renderEnv(self,xx=0,yy=0):
        #print(self.x,xx,self.x+xx,self.sizeS1)
        #print(self.y,yy,self.y+yy,self.sizeS2)
        #test = time.time()
        x = self.x + xx
        y = self.y + yy

        #a = np.zeros([self.win_size + 2, 4, 4]).astype(int)
        a = np.zeros([self.Z*(self.win_size+2),self.Z*4,4]).astype(int)

        if x+self.win_size > self.sizeS1:
            #print(self.win_size)
            #print(self.seq1[self.x:self.sizeS1])
            i = np.zeros([self.Z*(self.sizeS1-x),self.Z,4])
            for _ in range(self.Z):
                for __ in range(self.Z):
                    i[self.Z*np.arange(self.sizeS1-x)+__,_,np.array(self.seq1[x:self.sizeS1],dtype=int)]=1
            a[self.Z:self.Z+self.Z*(self.sizeS1-x),self.Z:2*self.Z,:]=i
            #a[self.Z*(self.sizeS1-x):,self.Z:2*self.Z,:]=1
        else:
            #print(self.win_size)
            #print(self.seq1[self.x:self.x+self.win_size])
            i = np.zeros([self.Z*(self.win_size),self.Z,4])
            for _ in range(self.Z):
                for __ in range(self.Z):
                    i[self.Z*np.arange(self.win_size)+__,_,np.array(self.seq1[x:x+self.win_size],dtype=int)]=1
            a[self.Z:-self.Z,self.Z:2*self.Z,:] = i

        if y+self.win_size > self.sizeS2:
            #print(self.win_size)
            #print(self.seq1[self.y:self.sizeS2])
            i = np.zeros([self.Z*(self.sizeS2-y),self.Z,4])
            for _ in range(self.Z):
                for __ in range(self.Z):
                    i[self.Z*np.arange(self.sizeS2-y)+__,_,np.array(self.seq2[y:self.sizeS2],dtype=int)]=1
            a[self.Z:self.Z+self.Z*(self.sizeS2-y),2*self.Z:3*self.Z,:]=i
            #a[self.Z*(self.sizeS2-y):,2*self.Z:3*self.Z,:]=1
        else:
            #print(self.win_size)
            #print(self.seq1[self.y:self.y+self.win_size])
            i = np.zeros([self.Z*(self.win_size),self.Z,4])
            for _ in range(self.Z):
                for __ in range(self.Z):
                    i[self.Z*np.arange(self.win_size)+__,_,np.array(self.seq2[y:y+self.win_size],dtype=int)]=1
            a[self.Z:-self.Z,2*self.Z:3*self.Z,:] = i

        #print("setup time :",time.time()-test)
        #test = time.time()

        r = (1-a[:,:,0])*(1-a[:,:,3])
        g = (1-a[:,:,1])*(1-a[:,:,3])
        b = (1-a[:,:,2])*(1-a[:,:,3])
        #print("RGB matching time :",time.time()-test)

        a = np.stack([r,g,b], axis=2)

        return a

    def renderRev(self,xx=0,yy=0):
        #print(self.x,xx,self.x+xx,self.sizeS1)
        #print(self.y,yy,self.y+yy,self.sizeS2)
        #test = time.time()
        x = self.x + xx
        y = self.y + yy

        #a = np.zeros([self.win_size + 2, 4, 4]).astype(int)
        a = np.zeros([self.Z*(self.win_size+2),self.Z*4,4]).astype(int)

        if x-self.win_size < 0:
            #print(self.win_size)
            #print(self.seq1[self.x:self.sizeS1])
            i = np.zeros([self.Z*(x+1),self.Z,4])
            for _ in range(self.Z):
                for __ in range(self.Z):
                    i[self.Z*np.arange(x+1)+__,_,np.array(self.seq1[0:x+1][::-1],dtype=int)]=1
            a[self.Z:self.Z+self.Z*(x+1),self.Z:2*self.Z,:]=i
            #a[self.Z*(self.sizeS1-x):,self.Z:2*self.Z,:]=1
        else:
            #print(self.win_size)
            #print(self.seq1[self.x:self.x+self.win_size])
            i = np.zeros([self.Z*(self.win_size),self.Z,4])
            for _ in range(self.Z):
                for __ in range(self.Z):
                    i[self.Z*np.arange(self.win_size)+__,_,np.array(self.seq1[x-self.win_size+1:x+1][::-1],dtype=int)]=1
            a[self.Z:-self.Z,self.Z:2*self.Z,:] = i

        if y-self.win_size < 0:
            #print(self.win_size)
            #print(self.seq1[self.y:self.sizeS2])
            i = np.zeros([self.Z*(y+1),self.Z,4])
            for _ in range(self.Z):
                for __ in range(self.Z):
                    i[self.Z*np.arange(y+1)+__,_,np.array(self.seq2[0:y+1][::-1],dtype=int)]=1
            a[self.Z:self.Z+self.Z*(y+1),2*self.Z:3*self.Z,:]=i
            #a[self.Z*(self.sizeS2-y):,2*self.Z:3*self.Z,:]=1
        else:
            #print(self.win_size)
            #print(self.seq1[self.y:self.y+self.win_size])
            i = np.zeros([self.Z*(self.win_size),self.Z,4])
            for _ in range(self.Z):
                for __ in range(self.Z):
                    i[self.Z*np.arange(self.win_size)+__,_,np.array(self.seq2[y-self.win_size+1:y+1][::-1],dtype=int)]=1
            a[self.Z:-self.Z,2*self.Z:3*self.Z,:] = i

        #print("setup time :",time.time()-test)
        #test = time.time()

        r = (1-a[:,:,0])*(1-a[:,:,3])
        g = (1-a[:,:,1])*(1-a[:,:,3])
        b = (1-a[:,:,2])*(1-a[:,:,3])
        #print("RGB matching time :",time.time()-test)

        a = np.stack([r,g,b], axis=2)

        return a

    def renderRC(self,xx=0,yy=0):
        #print(self.x,xx,self.x+xx,self.sizeS1)
        #print(self.y,yy,self.y+yy,self.sizeS2)
        #test = time.time()
        x = self.x + xx
        y = self.y + yy

        #a = np.zeros([self.win_size + 2, 4, 4]).astype(int)
        a = np.zeros([self.Z*(self.win_size+2),self.Z*4,4]).astype(int)

        if x+self.win_size > self.sizeS1:
            #print(self.win_size)
            #print(self.seq1[self.x:self.sizeS1])
            i = np.zeros([self.Z*(self.sizeS1-x),self.Z,4])
            for _ in range(self.Z):
                for __ in range(self.Z):
                    i[self.Z*np.arange(self.sizeS1-x)+__,_,np.array(self.seq1[x:self.sizeS1],dtype=int)]=1
            a[self.Z:self.Z+self.Z*(self.sizeS1-x),self.Z:2*self.Z,:]=i
            #a[self.Z*(self.sizeS1-x):,self.Z:2*self.Z,:]=1
        else:
            #print(self.win_size)
            #print(self.seq1[self.x:self.x+self.win_size])
            i = np.zeros([self.Z*(self.win_size),self.Z,4])
            for _ in range(self.Z):
                for __ in range(self.Z):

                    i[self.Z*np.arange(self.win_size)+__,_,np.array(self.seq1[x:x+self.win_size],dtype=int)]=1
            a[self.Z:-self.Z,self.Z:2*self.Z,:] = i

        if y-self.win_size < 0:
            #print(self.win_size)
            #print(self.seq1[self.y:self.sizeS2])
            i = np.zeros([self.Z*(y+1),self.Z,4])
            for _ in range(self.Z):
                for __ in range(self.Z):
                    i[self.Z*np.arange(y+1)+__,_,np.array(self.rev2[0:y+1][::-1],dtype=int)]=1
            a[self.Z:self.Z+self.Z*(y+1),2*self.Z:3*self.Z,:]=i
            #a[self.Z*(self.sizeS2-y):,2*self.Z:3*self.Z,:]=1
        else:
            #print(self.win_size)
            #print(self.seq1[self.y:self.y+self.win_size])
            i = np.zeros([self.Z*(self.win_size),self.Z,4])
            for _ in range(self.Z):
                for __ in range(self.Z):
                    i[self.Z*np.arange(self.win_size)+__,_,np.array(self.rev2[y-self.win_size+1:y+1][::-1],dtype=int)]=1
            a[self.Z:-self.Z,2*self.Z:3*self.Z,:] = i

        #print("setup time :",time.time()-test)
        #test = time.time()

        r = (1-a[:,:,0])*(1-a[:,:,3])
        g = (1-a[:,:,1])*(1-a[:,:,3])
        b = (1-a[:,:,2])*(1-a[:,:,3])
        #print("RGB matching time :",time.time()-test)

        a = np.stack([r,g,b], axis=2)

        return a

    def renderRCRev(self,xx=0,yy=0):
        #print(self.x,xx,self.x+xx,self.sizeS1)
        #print(self.y,yy,self.y+yy,self.sizeS2)
        #test = time.time()
        x = self.x + xx
        y = self.y + yy

        #a = np.zeros([self.win_size + 2, 4, 4]).astype(int)
        a = np.zeros([self.Z*(self.win_size+2),self.Z*4,4]).astype(int)

        if x-self.win_size < 0:
            #print(self.win_size)
            #print(self.seq1[self.x:self.sizeS1])
            i = np.zeros([self.Z*(x+1),self.Z,4])
            for _ in range(self.Z):
                for __ in range(self.Z):
                    i[self.Z*np.arange(x+1)+__,_,np.array(self.seq1[0:x+1][::-1],dtype=int)]=1
            a[self.Z:self.Z+self.Z*(x+1),self.Z:2*self.Z,:]=i
            #a[self.Z*(self.sizeS1-x):,self.Z:2*self.Z,:]=1
        else:
            #print(self.win_size)
            #print(self.seq1[self.x:self.x+self.win_size])
            i = np.zeros([self.Z*(self.win_size),self.Z,4])
            for _ in range(self.Z):
                for __ in range(self.Z):
                    i[self.Z*np.arange(self.win_size)+__,_,np.array(self.seq1[x-self.win_size+1:x+1][::-1],dtype=int)]=1
            a[self.Z:-self.Z,self.Z:2*self.Z,:] = i

        if y+self.win_size > self.sizeS2:
            #print(self.win_size)
            #print(self.seq1[self.y:self.sizeS2])
            i = np.zeros([self.Z*(self.sizeS2-y),self.Z,4])
            for _ in range(self.Z):
                for __ in range(self.Z):
                    i[self.Z*np.arange(self.sizeS2-y)+__,_,np.array(self.rev2[y:self.sizeS2],dtype=int)]=1
            a[self.Z:self.Z+self.Z*(self.sizeS2-y),2*self.Z:3*self.Z,:]=i
            #a[self.Z*(self.sizeS2-y):,2*self.Z:3*self.Z,:]=1
        else:
            #print(self.win_size)
            #print(self.seq1[self.y:self.y+self.win_size])
            i = np.zeros([self.Z*(self.win_size),self.Z,4])
            for _ in range(self.Z):
                for __ in range(self.Z):
                    i[self.Z*np.arange(self.win_size)+__,_,np.array(self.rev2[y:y+self.win_size],dtype=int)]=1
            a[self.Z:-self.Z,2*self.Z:3*self.Z,:] = i

        #print("setup time :",time.time()-test)
        #test = time.time()

        r = (1-a[:,:,0])*(1-a[:,:,3])
        g = (1-a[:,:,1])*(1-a[:,:,3])
        b = (1-a[:,:,2])*(1-a[:,:,3])
        #print("RGB matching time :",time.time()-test)

        a = np.stack([r,g,b], axis=2)

        return a

    def renderDiff(self,xx=0,yy=0):
        #print(self.x,xx,self.x+xx,self.sizeS1)
        #print(self.y,yy,self.y+yy,self.sizeS2)
        #test = time.time()
        x = self.x + xx
        y = self.y + yy

        #a = np.zeros([self.win_size + 2, 4, 4]).astype(int)
        a = np.zeros([self.Z*(self.win_size+2),self.Z*4,8]).astype(float)

        if x+self.win_size > self.sizeS1:
            #print(self.win_size)
            #print(self.seq1[self.x:self.sizeS1])
            i = np.zeros([self.Z*(self.sizeS1-x),self.Z,4])
            for _ in range(self.Z):
                for __ in range(self.Z):
                    i[self.Z*np.arange(self.sizeS1-x)+__,_,np.array(self.seq1[x:self.sizeS1],dtype=int)]=1
            a[self.Z:self.Z+self.Z*(self.sizeS1-x),self.Z:2*self.Z,:4]=i
            #a[self.Z*(self.sizeS1-x):,self.Z:2*self.Z,:]=1
        else:
            #print(self.win_size)
            #print(self.seq1[self.x:self.x+self.win_size])
            i = np.zeros([self.Z*(self.win_size),self.Z,4])
            for _ in range(self.Z):
                for __ in range(self.Z):
                    i[self.Z*np.arange(self.win_size)+__,_,np.array(self.seq1[x:x+self.win_size],dtype=int)]=1
            a[self.Z:-self.Z,self.Z:2*self.Z,:4] = i

        if y+self.win_size > self.sizeS2:
            #print(self.win_size)

            #print(self.seq1[self.y:self.sizeS2])
            i = np.zeros([self.Z*(self.sizeS2-y),self.Z,4])
            for _ in range(self.Z):
                for __ in range(self.Z):
                    i[self.Z*np.arange(self.sizeS2-y)+__,_,np.array(self.seq2[y:self.sizeS2],dtype=int)]=1
            a[self.Z:self.Z+self.Z*(self.sizeS2-y),2*self.Z:3*self.Z,:4]=i
            #a[self.Z*(self.sizeS2-y):,2*self.Z:3*self.Z,:]=1
        else:
            #print(self.win_size)
            #print(self.seq1[self.y:self.y+self.win_size])
            i = np.zeros([self.Z*(self.win_size),self.Z,4])
            for _ in range(self.Z):
                for __ in range(self.Z):
                    i[self.Z*np.arange(self.win_size)+__,_,np.array(self.seq2[y:y+self.win_size],dtype=int)]=1
            a[self.Z:-self.Z,2*self.Z:3*self.Z,:4] = i

        Kword1 = np.zeros([self.win_size,4]).astype(float)
        Kword2 = np.zeros([self.win_size,4]).astype(float)

        for i in range(self.win_size):	
            if x+(i+1)*self.K <= self.sizeS1:
                Kword1[i,0] = np.sum(self.seq1[x+i*self.K:x+(i+1)*self.K]==0)/self.K
                Kword1[i,1] = np.sum(self.seq1[x+i*self.K:x+(i+1)*self.K]==1)/self.K
                Kword1[i,2] = np.sum(self.seq1[x+i*self.K:x+(i+1)*self.K]==2)/self.K
                Kword1[i,3] = np.sum(self.seq1[x+i*self.K:x+(i+1)*self.K]==3)/self.K
            else:
                Kword1[i,0] = np.sum(self.seq1[x+i*self.K:self.sizeS1]==0)/self.K
                Kword1[i,1] = np.sum(self.seq1[x+i*self.K:self.sizeS1]==1)/self.K
                Kword1[i,2] = np.sum(self.seq1[x+i*self.K:self.sizeS1]==2)/self.K
                Kword1[i,3] = np.sum(self.seq1[x+i*self.K:self.sizeS1]==3)/self.K

            if y+(i+1)*self.K <= self.sizeS2:
                Kword2[i,0] = np.sum(self.seq2[y+i*self.K:y+(i+1)*self.K]==0)/self.K
                Kword2[i,1] = np.sum(self.seq2[y+i*self.K:y+(i+1)*self.K]==1)/self.K
                Kword2[i,2] = np.sum(self.seq2[y+i*self.K:y+(i+1)*self.K]==2)/self.K
                Kword2[i,3] = np.sum(self.seq2[y+i*self.K:y+(i+1)*self.K]==3)/self.K
            else:
                Kword2[i,0] = np.sum(self.seq2[y+i*self.K:y+self.sizeS2]==0)/self.K
                Kword2[i,1] = np.sum(self.seq2[y+i*self.K:y+self.sizeS2]==1)/self.K
                Kword2[i,2] = np.sum(self.seq2[y+i*self.K:y+self.sizeS2]==2)/self.K
                Kword2[i,3] = np.sum(self.seq2[y+i*self.K:y+self.sizeS2]==3)/self.K

        for _ in range(self.Z):
            for __ in range(self.Z):
                a[self.Z+self.Z*np.arange(self.win_size)+__,self.Z+_,4:8] = Kword1
                a[self.Z+self.Z*np.arange(self.win_size)+__,2*self.Z+_,4:8] = Kword2

        #print(a)
        return a

    def renderFFT(self,xx=0,yy=0):
        #test = time.time()
        x = self.x + xx
        y = self.y + yy

        a = np.zeros([self.freq_size,4,4]).astype(float)

        if x+self.win_size > self.sizeS1:
            i = np.zeros([self.win_size,4]).astype(float)
            i[np.arange(self.sizeS1-x),np.array(self.seq1[x:self.sizeS1],dtype=int)]=1
            for j in range(4):
                fftval = np.fft.fft(i[:,j],self.freq_size)
                a[:,0,j]=np.real(fftval)
                a[:,1,j]=np.imag(fftval)
        else:
            i = np.zeros([self.win_size,4]).astype(float)
            i[np.arange(self.win_size),np.array(self.seq1[x:x+self.win_size],dtype=int)]=1
            for j in range(4):
                fftval = np.fft.fft(i[:,j],self.freq_size)
                a[:,0,j]=np.real(fftval)
                a[:,1,j]=np.imag(fftval)

        if y+self.win_size > self.sizeS2:
            i = np.zeros([self.win_size,4]).astype(float)
            i[np.arange(self.sizeS2-y),np.array(self.seq2[y:self.sizeS2],dtype=int)]=1
            for j in range(4):
                fftval = np.fft.fft(i[:,j],self.freq_size)
                a[:,2,j]=np.real(fftval)
                a[:,3,j]=np.imag(fftval)
        else:
            i = np.zeros([self.win_size,4]).astype(float)
            i[np.arange(self.win_size),np.array(self.seq2[y:y+self.win_size],dtype=int)]=1
            for j in range(4):
                fftval = np.fft.fft(i[:,j],self.freq_size)
                a[:,2,j]=np.real(fftval)
                a[:,3,j]=np.imag(fftval)

        #print(time.time()-test)
        return a

    def step(self, action):
        reward, done = self.moveChar(action)
        state = self.renderEnv()
        return state, reward, done

    def teststep(self, action):
        reward, done = self.moveChar(action)
        return reward, done

    def stepRC(self, action):
        reward, done = self.moveCharRC(action)
        return reward, done

    def stepDiff(self, action):
        reward, done = self.moveChar(action)
        state = self.renderDiff()
        return state, reward, done

    def stepFFT(self, action):
        reward, done = self.moveChar(action)
        state = self.renderFFT()
        return state, reward, done
