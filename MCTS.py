#Une implémentation rapide de Monte Carlo Tree Search.
# le "State" est juste un jeu où vous avez NUM_TURNS et au tour i vous pouvez faire un choix de [-2,2,3,-3] * i et ce à une valeur cumulée. L'objectif est que la valeur cumulée soit aussi proche de 0 que possible.
#En particulier, il existe deux modèles de "best child" que l'on peut utiliser


#Pour executer le programme :$python MCTS.py --num_simulationss 10000 --levels 10


#Librairie
import random
import math
import argparse

sc=1/math.sqrt(2)

class State():
	TURNS = 15	
	GOAL = 0
	MOVES=[-2,2,3,-3]
	MAX_VAL= (5.0*(TURNS-1)*TURNS)/2
	num_moves=len(MOVES)

    #Initialiseur 
	def __init__(self, value=0, moves=[], turn=TURNS):
		self.value=value
		self.turn=turn
		self.moves=moves

    #Fonction qui retourne l'action suivante
	def next_state(self):
		nextmove=random.choice([x*self.turn for x  in self.MOVES])
		next=State(self.value+nextmove, self.moves+[nextmove],self.turn-1)
		return next

    #Fonction qui retourne l'action terminale
	def terminal(self):
		if self.turn == 0:
			return True
		return False

    #Fonction qui retourne l'action récompense
	def reward(self):
		r = 1.0-(abs(self.value-self.GOAL)/self.MAX_VAL)
		return r

    #Fonction qui retourne l'action representation ecrite du resultat
	def __repr__(self):
		s="Value: %d; Moves: %s"%(self.value,self.moves)
		return s
	

class Node():

    #Initialiseur 
	def __init__(self, state, parent=None):
		self.visits=1
		self.reward=0.0	
		self.state=state
		self.children=[]
		self.parent=parent	

    #Fonction qui construit node child
	def add_child(self,child_state):
		child=Node(child_state,self)
		self.children.append(child)

    #Fonction qui fait la mise à jour de la récompense
	def update(self,reward):
		self.reward+=reward
		self.visits+=1

    #Fonction qui retourne si l'expansion est arrivée à sa limite ou non
	def fully_expanded(self):
		if len(self.children)==self.state.num_moves:
			return True
		return False
    #Fonction qui retourne l'action representation ecrite du resultat
	def __repr__(self):
		s="Node; children: %d; visits: %d; reward: %f"%(len(self.children),self.visits,self.reward)
		return s
		

#UCT algorithm (Upper Confidence bounds applied to Trees)
def UCT(b,root):
	for i in range(int(b)):
		front=POLICY(root)
		reward=DEFAULTPOLICY(front.state)
		BACKUP(front,reward)
	return BESTCHILD(root,0)

#Fonction qui retourne la mise a jour de la récompense 
def DEFAULTPOLICY(state):
	while state.terminal()==False:
		state=state.next_state()
	return state.reward()

#Politique de l'arbre ,comment se fait l'exploitation de l'arbre
def POLICY(node):
	while node.state.terminal()==False:
		if len(node.children)==0:
			return EXPAND(node)
		elif random.uniform(0,1)<.5:
			node=BESTCHILD(node,sc)
		else:
			if node.fully_expanded()==False:	
				return EXPAND(node)
			else:
				node=BESTCHILD(node,sc)
	return node

#Fonction qui permet l'expansion de l'arbre
def EXPAND(node):
	tried_children=[c.state for c in node.children]
	new_state=node.state.next_state()
	while new_state in tried_children:
		new_state=node.state.next_state()
	node.add_child(new_state)
	return node.children[-1]

#Fonction qui retourne le "Best Child"
def BESTCHILD(node,scalar):
	bestscore=0.0
	bestchildren=[]
	for c in node.children:
		exploit=c.reward/c.visits
		explore=math.sqrt(2.0*math.log(node.visits)/float(c.visits))	
		score=exploit+scalar*explore
		if score==bestscore:
			bestchildren.append(c)
		if score>bestscore:
			bestchildren=[c]
			bestscore=score
	return random.choice(bestchildren)


#Fonction qui permet de faire les MAJ lors de l'UCT
def BACKUP(node,reward):
	while node!=None:
		node.visits+=1
		node.reward+=reward
		node=node.parent
	return


if __name__=="__main__":
	parser = argparse.ArgumentParser(description='MCTS research code')
	parser.add_argument('--num_simulations', action="store", required=True, type=int)
	parser.add_argument('--levels', action="store", required=True, type=int, choices=range(State.NUM_TURNS))
	args=parser.parse_args()
	
	current_node=Node(State())
	for l in range(args.levels):
		current_node=UCT(args.num_simulations/(l+1),current_node)
		print("level %d"%l)
		print("Num Children: %d"%len(current_node.children))
		for i,c in enumerate(current_node.children):
			print(i,c)
		print("Best Child: %s"%current_node.state)
		
			
	