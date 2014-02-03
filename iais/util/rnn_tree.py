'''
Created on Oct 19, 2013

@author: bhanu
'''

from nltk.tree import Tree
import treelib

class MVRNNTree(object):
    '''
    Tree structure for RNN
    '''
    def __init__(self, nltk_tree):
        ''' create a tree for mvrnn network using a parsed nltk tree '''
        self.tree = treelib.Tree()
        self.nNodes = 0
        self.totalScore = None   
        root = MVRNNNode(identifier=self.nNodes)   
        self.tree.add_node(root)  
        self._create_tree(root, nltk_tree)
    
        
    def _create_tree(self, parent, tree):
        assert tree != None         
        if(not (tree.node == 'ROOT' and parent.nodeid==0)):  
            self.nNodes += 1           
            node = MVRNNNode(identifier=self.nNodes)            
            self.tree.add_node(node, parent.nodeid)
            parent = node                 
        for child in tree:
            self._create_tree(parent, child)
            


class MVRNNNode(treelib.Node):
    
    def __init__(self, tag=None, identifier=None, expanded=True):
        self.nodeid = identifier
        self.label = None
        self.features = None
        self.matrix = None
        treelib.Node.__init__(self, tag=None, identifier=None, expanded=True)
                