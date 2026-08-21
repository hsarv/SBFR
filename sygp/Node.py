import numpy as np
import pydot
from IPython.display import Image
import pandas as pd
from math import log

import warnings
import os
os.environ["PATH"] += os.pathsep + "C:\\Program Files\\Graphviz\\bin"
warnings.filterwarnings("ignore")

# 
# By using this file, you are agreeing to this product's EULA
#
# This product can be obtained in https://github.com/jespb/Python-M3GP
#
# Copyright ©2019-2022 J. E. Batista
#

class Node:
	branches = None
	value = None


	def __init__(self):

		pass

	def setBranche(self,node,i):
		if i==0:
			self.branches=[]
		self.branches.append(node)

	def setValue(self,val):
		self.value=val

	def create(self, rng, operators=None, terminals=None, depth=None,full=False):
		if depth>1 and (rng.random()<0.5 or full ==True ):
			op, n_args = operators[rng.randint(0,len(operators)-1)]
			self.value = op

			self.branches = []
			for i in range(n_args):
				n = Node()
				n.create(rng, operators, terminals, depth-1)
				self.branches.append(n)
		else:
			df_terminals = pd.DataFrame(columns=terminals)
			new_terminal=df_terminals.filter(regex='^F').columns.tolist()
			old_termonals=df_terminals.columns.difference(new_terminal)

			probability=0.5
			if rng.random()<probability or len(new_terminal)<1:
				self.value = old_termonals[rng.randint(0,len(old_termonals)-1)]
			else:
				self.value = new_terminal[rng.randint(0, len(new_terminal) - 1)]

	def createCombNode(self, rng, operators=None, terminals=None, depth=None,full=False):
		if depth>1 and (rng.random()<0.5 or full ==True ):
			op, n_args = operators[rng.randint(0,len(operators)-1)]
			self.value = op

			self.branches = []
			for i in range(n_args):
				n = Node()
				n.create(rng, operators, terminals, depth-1)
				self.branches.append(n)
		else:
			self.value = terminals[rng.randint(0,len(terminals)-1)] # Sem literais

	def createTerminals(self, operators=None, terminals=None, depth=1, full=False, i=0):
		self.value = terminals[i]

	def copy(self,value=None, branches=None):
		self.branches = branches
		self.value=value


	def __str__(self):
		if self.branches == None:
			return str(self.value)
		else:
			if len(self.branches) == 2:
				return "( " + str(self.branches[0]) + " " + str(self.value) + " " + str(self.branches[1]) + " )"
			else:
				return str(self.value) + " ( " + " ".join( [str(b) for b in self.branches] ) + " )"


	def getSize(self):
		'''
		Returns the total number of nodes within this Node.
		'''
		if self.branches == None:
			return 1
		else:

			return 1 + sum( [b.getSize() for b in self.branches] )


	def getDepth(self):
		'''
		Returns the depth of this Node.
		'''
		if self.branches == None:
			return 1
		else:
			return 1 + max( [b.getDepth() for b in self.branches] )


	def getRandomNode(self, rng, value=None):
		'''
		Returns a random Node within this Node.
		'''
		if value == None:
			if self.getSize()-1==0:
				value=0
			else:
				if isinstance(rng, np.random._generator.Generator):
					value= rng.integers(0,self.getSize()-1)  # Use 'integers' for the new RNG
				else:
					value= rng.randint(0,self.getSize()-1)
		if value == 0:
			#print(self)
			return self

		#print(value, self)
		for i in range(len(self.branches)):
			size = self.branches[i].getSize()
			if value-1 < size:
				return self.branches[i].getRandomNode(rng, value-1)
			value -= size


	def swap(self, other):
		'''
		Swaps the content of two nodes.
		'''
		b = self.branches
		v = self.value

		self.branches = other.branches
		self.value = other.value

		other.branches = b
		other.value = v


	def clone(self):
		'''
		Returns a clone of this node.
		'''
		if self.branches == None:
			n = Node()
			n.copy(value=self.value, branches = None)
			return n
		else:
			n = Node()
			n.copy(value=self.value, branches=[b.clone() for b in self.branches])
			return n

	def safe_output(self,arr):
		arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)  # replace NaN/Inf
		arr = np.clip(arr, -1e6, 1e6)  # keep numbers in a safe range
		return arr

	def calculate(self, sample):
		'''
        Returns the calculated value of a sample.
        '''
		if self.branches is None:
			try:
				result = np.array(sample[self.value])  # variable
			except:
				result = np.array([float(self.value)] * sample.shape[0])  # constant
			return self.safe_output(result)

		if self.value == "+":  # +
			result = self.branches[0].calculate(sample) + self.branches[1].calculate(sample)

		elif self.value == "-":  # -
			result = self.branches[0].calculate(sample) - self.branches[1].calculate(sample)

		elif self.value == "*":  # *
			result = self.branches[0].calculate(sample) * self.branches[1].calculate(sample)

		elif self.value == "/":  # /
			right = self.branches[1].calculate(sample)
			right = np.where(right == 0, 1, right)  # avoid division by zero
			result = self.branches[0].calculate(sample) / right

		elif self.value == "log2":  # log2(X)
			res = self.branches[0].calculate(sample)
			res = np.where(res <= 0, 1, res)
			result = np.log2(res)

		elif self.value == "log":  # natural log
			res = self.branches[0].calculate(sample)
			res = np.where(res <= 0, 1, res)
			result = np.log(res)

		elif self.value == "exp":  # exp(X)
			res = self.branches[0].calculate(sample)
			result = np.exp(res)

		elif self.value == "pow":  # power(base, exponent)
			base = self.branches[0].calculate(sample)
			expn = self.branches[1].calculate(sample)
			safe_base = np.where(base < 0, 0, base)  # avoid invalid operations
			result = np.power(safe_base, expn)

		elif self.value == "max":  # max(X0, X1, ... Xn)
			calc = [b.calculate(sample) for b in self.branches]
			result = np.max(np.vstack(calc), axis=0)

		elif self.value == "min":  # min(X0, X1, ... Xn)
			calc = [b.calculate(sample) for b in self.branches]
			result = np.min(np.vstack(calc), axis=0)

		else:
			raise ValueError(f"Unsupported operator: {self.value}")

		return self.safe_output(result)

	def isLeaf(self):
		'''
		Returns True if the Node had no sub-nodes.
		'''
		return self.branches == None

	def getSemantics(self,tr_x):
		'''
		Returns the semantic of a Node.
		'''		
		return self.calculate(tr_x)

	def redirect(self, other):
		'''
		Assigns the content of another Node to this Node.
		'''
		self.value = other.value
		self.branches = other.branches

	def visualize_tree(self,node,name):
		# Create a new pydot graph
		graph = pydot.Dot(graph_type='graph')

		# Recursive function to add nodes and edges to the graph
		def add_nodes_edges(node, graph):
			# Create a node for the current value
			graph_node = pydot.Node(id(node), label=node.value)
			graph.add_node(graph_node)

			# Check if branches is not None, and then add them recursively
			if node.branches is not None:
				for branch in node.branches:
					branch_node = add_nodes_edges(branch, graph)
					graph.add_edge(pydot.Edge(graph_node, branch_node))

			return graph_node

		# Start adding nodes and edges from the root node
		add_nodes_edges(node, graph)

		# Render the graph to a PNG image and display it
		graph.write_png(name)
		return Image(filename=name)

	def prun(self, tr_x):
		'''
		Simplifies this Node
		'''
		semantics = self.getSemantics(tr_x)
		semantics.sort()
		if semantics[0]== semantics[-1] and len(semantics)>1:
			self.value = str(semantics[0])
			self.branches = None



		if self.branches!=None and len(self.branches)==1: # [log2]
			pass


		
		if self.branches!=None and len(self.branches)==2: # [+, -, *, /]
			# +
			if self.value == "+":
				# 0 + X == X
				if not self.isLeaf() and ( self.branches[0].isLeaf() and self.branches[0].value == "0.0" ):
					self.redirect(self.branches[1])

				# X + 0 == X
				if not self.isLeaf() and ( self.branches[1].isLeaf() and self.branches[1].value == "0.0" ):
					self.redirect(self.branches[0])

				# X + X == 2 * X
				if not self.isLeaf() and ( str(self.branches[1]) == str(self.branches[0]) ):
					self.value = "*"
					n = Node()
					n.copy(value = "2.0")
					self.branches[0].redirect( n )

			# - 
			if self.value == "-":
				# X - 0 == X
				if not self.isLeaf() and ( self.branches[1].isLeaf() and self.branches[1].value == "0.0" ):
					self.redirect(self.branches[0])

				# X - X == 0
				if not self.isLeaf() and ( str(self.branches[1]) == str(self.branches[0]) ):
					n = Node()
					n.copy(value = "0.0")
					self.redirect( n )

			# * 
			if self.value == "*":
				# X * 0 == 0,  0 * X == 0
				if not self.isLeaf() and ( (self.branches[0].isLeaf() and self.branches[0].value=="0.0") or (self.branches[1].isLeaf() and self.branches[1].value=="0.0") ):
					n = Node()
					n.copy(value = "0.0")
					self.redirect( n )

				# 1 * X == X
				if not self.isLeaf() and ( self.branches[0].isLeaf() and self.branches[0].value == "1.0" ):
					self.redirect(self.branches[1])

				# X * 1 == X
				if not self.isLeaf() and ( self.branches[1].isLeaf() and self.branches[1].value == "1.0" ):
					self.redirect(self.branches[0])

			# //
			if self.value == "/":
				# X // 0 == 1
				if not self.isLeaf() and ( self.branches[1].isLeaf() and self.branches[1].value=="0.0" ):
					n = Node()
					n.copy(value = "1.0")
					self.redirect( n )

				# X // 1 == X
				if not self.isLeaf() and ( self.branches[1].isLeaf() and self.branches[1].value=="1.0" ):
					self.redirect(self.branches[0])

				# X // X == 1
				if not self.isLeaf() and ( str(self.branches[1]) == str(self.branches[0]) ):
					n = Node()
					n.copy(value = "1.0")
					self.redirect( n )


		if self.branches!=None and len(self.branches)==3: # [max]
			pass




		if self.branches != None:
			for branch in self.branches:
				branch.prun(tr_x)
