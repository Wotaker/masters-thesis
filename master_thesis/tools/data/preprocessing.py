from typing import Optional, Tuple, List, Callable

import numpy as np
import networkx as nx

class Preprocessing():

    def __init__(self) -> Callable[[np.ndarray], List[nx.DiGraph]]:
        
        return self.__call__

    def __call__(self) -> Tuple[List[nx.DiGraph], List[int]]:
        pass