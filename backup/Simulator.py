import os
import csv
import pandas
import numpy as np

# cacheline
class cacheline:
	def __init__(self,s,a):
		self.sector = s
		self.access = a

	def getsector(self):
		return self.sector

	def getaccess(self):
		return self.access


# a cache
class cache():
	def __init__(self,size):
		# cache size
		self.size = size
		# caches
		self.number = 0
		# miss times
		self.miss = 0
		# hit times
		self.hit = 0
		# clock pointer
		self.clock = 0
		# cache blocks 
		self.blocks = []

	# caculate the miss rate
	def get_missrate(self):
		return self.miss/(self.hit+self.miss)

	# find if the sector is cached for fifo
	def find_sector_fifo(self,sector):
		for i in self.blocks:
			if sector == i.getsector():
				self.hit = self.hit + 1
				return True
	
		self.miss = self.miss + 1
		return False
	
	# find if the sector is cached for LRU
	def find_sector_lru(self,sector):
		for i in self.blocks:
			if sector == i.getsector():
				self.hit = self.hit + 1
				self.blocks.append(i)
				self.blocks.remove(i)
				return True

		self.miss = self.miss + 1
		return False

	# find if the sector is cached for MRU
	def find_sector_mru(self,sector):
		for i in self.blocks:
			if sector == i.getsector():
				self.hit = self.hit + 1
				self.blocks.append(i)
				self.blocks.remove(i)
				return True
		self.miss = self.miss + 1
		return False

	# find if the sector is chached for clock
	def find_sector_clock(self,sector):
		for i in self.blocks:
			if sector == i.getsector():
				i.access = 1
				self.hit = self.hit + 1
				return True
		self.miss = self.miss + 1
		return False

	# fifo_al
	def fifo(self,sector):
		# if in cache
		if self.find_sector_fifo(sector) == True:
			return 

		# if not full, append to the end of the list
		if self.number < self.size:
			self.blocks.append(cacheline(sector,0))
			self.number = self.number + 1
			return 
		else:
			del self.blocks[0]
			self.blocks.append(cacheline(sector,0))
			return
	
	# LRU_al
	def lru(self,sector):
		# if in cache
		if self.find_sector_lru(sector) == True:
			return

		# if not full, append to the end of the list
		if self.number < self.size:
			self.blocks.append(cacheline(sector,0))
			self.number = self.number + 1
			return 
		else:
			del self.blocks[0]
			self.blocks.append(cacheline(sector,0))
			return

	# MRU_al
	def mru(self,sector):
		# if in cache
		if self.find_sector_mru(sector) == True:
			return

		# if not full
		if self.number < self.size:
			self.blocks.append(cacheline(sector,0))
			self.number = self.number + 1
			return
		else:
			self.blocks.pop()
			self.blocks.append(cacheline(sector,0))
			return

	# radom_al
	def rdm(self,sector):
		# if in cache
		if self.find_sector_fifo(sector) == True:
			return 

		# if not full, append to the end of the list
		if self.number < self.size:
			self.blocks.append(cacheline(sector,0))
			self.number = self.number + 1
			return 
		else:
			self.blocks[random.randint(0,self.size-1)] = cacheline(sector,0)
			return

	# clock_al
	def clk(self,sector):
		# if in cache
		if self.find_sector_clock(sector) == True:
			return

		# if not full, append to the end of the list
		if self.number < self.size:
			self.blocks.append(cacheline(sector,0))
			self.number = self.number + 1
			return 
		else:
			for i in self.blocks[self.clock:]:
				if i.getaccess() == 0:
					if self.clock == self.size-1:
						self.clock = 0
						self.blocks.remove(i)
						self.blocks.insert(self.size-1,cacheline(sector,0))
						return
					else:
						self.clock = self.blocks.index(i) + 1
						self.blocks.remove(i)
						self.blocks.insert(self.clock-1,cacheline(sector,0))
						return
				else:
					i.access = 0
			for i in self.blocks[:self.clock]:
				if i.getaccess() == 0:
					if self.clock == self.size-1:
						self.clock = 0
						self.blocks.remove(i)
						self.blocks.insert(self.size-1,cacheline(sector,0))
						return
					else:
						self.clock = self.blocks.index(i) + 1
						self.blocks.remove(i)
						self.blocks.insert(self.clock-1,cacheline(sector,0))
						return
				else:
					i.access = 0

			del self.blocks[self.clock]
			self.blocks.insert(self.clock,cacheline(sector,0))
			
			if self.clock == self.size-1:
				self.clock = 0
			else:
				self.clock = self.clock + 1 
			return

# file directory 
fd = r'/home/rio/Desktop/AI/data'

# create a list of all needed files in fd
def eachfile(fd):
	outputfile = []
	path = os.listdir(fd)

	headers = ["TESTNAME","FIFO","LRU","MRU","CLOCK","RANDOM"]
	result = open("result.csv", 'w', newline = '')
	result_csv = csv.writer(result)
	result_csv.writerow(headers)

	for file in path:
		# add the file to the path
		newdir = os.path.join(fd,file)

		# choose the file named ".csv" 
		if os.path.splitext(newdir)[1] == ".csv":
			testname = os.path.splitext(os.path.split(newdir)[1])[0]
			df = pandas.read_csv(newdir)

			test1,test2,test3,test4,test5 = cache(64),cache(64),cache(64),cache(64),cache(64)

			for i in df["blocksector"]:
				test1.fifo(i)
				test2.lru(i)
				test3.mru(i)
				test4.clk(i)
				test5.rdm(i)

			result_csv.writerow([testname,test1.get_missrate(),test2.get_missrate(),test3.get_missrate(),test4.get_missrate(),test5.get_missrate()])
			print(testname + " is finished")

eachfile(fd)