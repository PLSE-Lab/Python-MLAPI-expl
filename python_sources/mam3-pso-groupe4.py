import numpy as np
import numpy.random as rnd
import numpy.linalg as LA
import matplotlib.pyplot as plt


rng = rnd.default_rng()  # notre générateur d'aléatoire

# Notons que l'on utilise des classes pour ne pas déclarer des variables globales partout et pouvoir réutiliser
# les variables d'instance à travers les fonctions


class agent_PSO:
    '''[Classe qui décrit une particule d'un essaim]'''

    def __init__(self, n, starting_pos, min_bounds, max_bounds):
        '''[Constructeur d'une particule]


        Arguments:
            n {[int]} -- la dimension du modèle dans lequel cette particule sert
            starting_pos {[un vecteur de taille nx1]} -- [la position initiale de la particule]
            min_bounds {[un vecteur de taille nx1]} -- [contient les bornes en-dessous desquelles ne peut passer chaque coordonnée]
            max_bounds {[un vecteur de taille nx1]} -- [contient les bornes au-dessus desquelles ne peut passer chaque coordonnée]
        '''
        self.dim = n
        self.coords = starting_pos
        self.pb_coords = self.coords
        self.min_bounds = min_bounds
        self.max_bounds = max_bounds
        self.velocity = np.zeros((n, 1))
        self.neighbors = []

    def print(self):
        print((self.coords, self.pb_coords, self.velocity))

    def update_pos(self, new_velocity):
        self.velocity = new_velocity
        for i in range(self.dim):
            # la nouvelle position n'est que candidate, il reste a verifier qu'elle ne fait pas sortir la particule des bornes
            new_coords_candidate = self.coords + self.velocity
        self.coords = np.maximum(np.minimum(new_coords_candidate, self.max_bounds), self.min_bounds)

    def update_pb(self, new_pb_value):
        self.pb_coords = self.coords
        self.value_at_pb = new_pb_value

## Version optimisée pour la vitesse de convergence (par rapport à la classe modele_PSO) : n'implémente que le global best, et pas de topologie particulière

class model_SPSO:
    def __init__(self, nb_agents, dim, objective_function, min_bounds, max_bounds, phi1, phi2, inertia_from_to_step, known_solution=None, precision=10**(-5), max_cost=10**6):
        '''[Construit un modèle de PSO Standard pour des valeurs des paramètres particulières]

        Arguments:
            nb_agents {[int]} -- [le nombre de particules]
            dim {[int]} -- [la dimension de l'espace de depart de la fonction objectif]
            objective_function {[function]} -- [la fonction à optimiser]
            min_bounds {[un vecteur dimx1]} -- [le vecteur des bornes minimales de recherche]
            max_bounds {[un vecteur dimx1]} -- [le vecteur des bornes maximales de recherche]
            phi1 {[float]} -- [poids de l'influence personnelle]
            phi2 {[float]} -- [poids de l'influence sociale]
            inertia_from_to_step {[[float, float, float]]} -- [le max, min et pas de descente de l'inertie]

        Keyword Arguments:
            known_solution {[vecteur dimx1]} -- [solution du problème d'optimisation. If None, on ne teste pas sur la précision] (default: {None})
            precision {[float]} -- [precision desirée, doit être adaptée en fonction de la fonction testée] (default: {10**(-5)})
            max_cost {[int]} -- [nombre limite d'évaluations de la fonction objectif] (default: {10**6})
            will_store {bool} -- [si l'on doit ou non stocker les coordonnées des points au fil du temps] (default: {False})
        '''

        self.min_bounds = min_bounds
        self.max_bounds = max_bounds
        self.nb_agents = nb_agents
        self.dim = dim
        self.objective_function = objective_function
        self.phi1 = phi1
        self.phi2 = phi2
        self.inertia_prop = inertia_from_to_step
        self.inertia = self.inertia_prop[0]
        self.known_solution = known_solution
        self.precision = precision
        self.current_cost = 0
        self.max_cost = max_cost

        self.all_agents = self.create_agents()
        self.best_agent = sorted(self.all_agents, key=lambda a: a.value_at_pb)[0]

    def create_agents(self):
        '''[Fonction qui permet d'initialiser les particules]

        Returns:
            [list<agent_PSO>] -- [la liste de tous les agents du modèle]
        '''
        agents = []
        for i in range(self.nb_agents):
            # la position de départ de l'agent est un vecteur aléatoire entre les bornes et de taile dimx1
            new_agent = agent_PSO(self.dim, rng.uniform(self.min_bounds, self.max_bounds, (self.dim, 1)), self.min_bounds, self.max_bounds)
            new_agent.update_pb(self.objective_function(new_agent.coords))
            self.current_cost += 1

            agents.append(new_agent)

        return agents

    def iterate_all(self):
        '''[fonction qui effectue toutes les itérations du modèle]

        [s'arrête soit par précision atteinte soit par coût maximal atteint]
        '''
        while (self.current_cost < self.max_cost):
            # on trie les agents par leur meilleure valeur pour obtenir le meilleur d'entre eux
            self.iterate_step()
            self.best_agent = sorted(self.all_agents, key=lambda a: a.value_at_pb)[0]

            if self.known_solution is not None:  # vérification précision
                if LA.norm(self.best_agent.pb_coords - self.known_solution) < self.precision:
                    return (self.best_agent.pb_coords, self.best_agent.value_at_pb, "Precision")

        return (self.best_agent.pb_coords, self.best_agent.value_at_pb, "Cout")

    def iterate_step(self):
        '''[effectue 1 etape de l'algorithme]'''
        for agent in self.all_agents:
            agent.update_pos(self.compute_velocity(agent))
            current_value = self.objective_function(agent.coords)
            self.current_cost += 1
            if current_value < agent.value_at_pb:
                agent.update_pb(current_value)

        self.inertia = np.maximum(self.inertia_prop[1], self.inertia - self.inertia_prop[2])

    def compute_velocity(self, agent):
        '''[Calcul de la vitesse d'une particule]
        Arguments:
            agent {[agent_PSO]} -- [l'agent dont on veut la vitesse]

        Returns:
            [un vecteur dimx1] -- [vecteur vitesse]
        '''

        return self.inertia * agent.velocity + self.phi1 * rng.uniform(0, 1) * (agent.pb_coords - agent.coords) + self.phi2 * rng.uniform(0, 1) * (self.best_agent.coords - agent.coords)

    def reset(self):
        self.inertia = self.inertia_prop[0]
        self.current_cost = self.nb_agents
        self.all_agents = self.create_agents()
        self.best_agent = sorted(self.all_agents, key=lambda a: a.value_at_pb)[0]


def schaffer2(x, y):
    return 0.5 + (np.sin(x**2 - y**2)**2 - 0.5) / ((1 + 0.001 * (x**2 + y**2))**2)


def schaffer2_vect(coords):
    return schaffer2(coords[0], coords[1])


modele_standard = model_SPSO(nb_agents=70,
                    dim=2,
                    objective_function=schaffer2_vect,
                    min_bounds=np.array([[-100], [-100]]),
                    max_bounds=np.array([[100], [100]]),
                    phi1=0.2,
                    phi2=2.0,
                    inertia_from_to_step=[0.8, 0.1, 0.0005],
                    known_solution=np.array([[0], [0]]),
                    precision=10**(-5),
                    max_cost=10**6)

pos, val, motif = modele_standard.iterate_all()
print("Position de la meilleure particule:\n", pos)
print("Valeur en ce point:\n", val)
print("Motif d'arrêt: ", motif)

## Version avec choix de topologie (et implémentation du stockage pour un éventuel affichage)
# Les différences principales résident dans la matrice de communication et la méthode get_social

class model_PSO:
    def __init__(self, nb_agents, dim, objective_function, min_bounds, max_bounds, phi1, phi2, communication_matrix, inertia_from_to_step, known_solution=None, precision=10**(-5), max_cost=10**6, will_store=False):
        '''[Construit un modèle de PSO pour des valeurs des paramètres particulières]

        Arguments:
            nb_agents {[int]} -- [le nombre de particules]
            dim {[int]} -- [la dimension de l'espace de depart de la fonction objectif]
            objective_function {[function]} -- [la fonction à optimiser]
            min_bounds {[un vecteur dimx1]} -- [le vecteur des bornes minimales de recherche]
            max_bounds {[un vecteur dimx1]} -- [le vecteur des bornes maximales de recherche]
            phi1 {[float]} -- [poids de l'influence personnelle]
            phi2 {[float]} -- [poids de l'influence sociale]
            communication_matrix {[matrice carrée de taille nb_agents]} -- [Matrice qui décrit la topologie sociale des particules: M[i, j]==1 ssi les particules i et j peuvent communiquer]
            inertia_from_to_step {[[float, float, float]]} -- [le max, min et pas de descente de l'inertie]

        Keyword Arguments:
            known_solution {[vecteur dimx1]} -- [solution du problème d'optimisation. If None, on ne teste pas sur la précision] (default: {None})
            precision {[float]} -- [precision desirée, doit être adaptée en fonction de la fonction testée] (default: {10**(-5)})
            max_cost {[int]} -- [nombre limite d'évaluations de la fonction objectif] (default: {10**6})
            will_store {bool} -- [si l'on doit ou non stocker les coordonnées des points au fil du temps] (default: {False})
        '''

        self.min_bounds = min_bounds
        self.max_bounds = max_bounds
        self.nb_agents = nb_agents
        self.dim = dim
        self.objective_function = objective_function
        self.phi1 = phi1
        self.phi2 = phi2
        self.communication_matrix = communication_matrix
        self.inertia_prop = inertia_from_to_step
        self.inertia = inertia_from_to_step[0]
        self.known_solution = known_solution
        self.precision = precision
        self.current_cost = 0
        self.max_cost = max_cost

        self.all_agents = self.create_agents()

        # remplissage des voisins
        for i in range(self.nb_agents):
            for j in range(self.nb_agents):
                if self.communication_matrix[i, j] == 1:
                    # La matrice de communication n'a pas à être symétrique, on peut imaginer des communications non réciproques
                    self.all_agents[i].neighbors.append(self.all_agents[j])

        self.will_store = will_store
        if self.will_store:
            # declaration de la matrice de stockage, qui est plus grande que nécessaire pour avoir de la marge
            self.all_coords = np.zeros(((self.max_cost // self.nb_agents) + 1, self.nb_agents, self.dim, 1))
            self.all_coords[0] = np.array([a.coords for a in self.all_agents])

    def create_agents(self):
        '''[Fonction qui permet d'initialiser les particules]

        Returns:
            [list<agent_PSO>] -- [la liste de tous les agents du modèle]
        '''
        agents = []
        for i in range(self.nb_agents):
            # la position de départ de l'agent est un vecteur aléatoire entre les bornes et de taile dimx1
            new_agent = agent_PSO(self.dim, rng.uniform(self.min_bounds, self.max_bounds, (self.dim, 1)), self.min_bounds, self.max_bounds)
            new_agent.update_pb(self.objective_function(new_agent.coords))
            self.current_cost += 1

            agents.append(new_agent)

        return agents

    def iterate_all(self):
        '''[fonction qui effectue toutes les itérations du modèle]

        [s'arrête soit par précision atteinte soit par coût maximal atteint]
        '''
        self.step = 0
        while (self.current_cost < self.max_cost):
            # on trie les agents par leur meilleure valeur pour obtenir le meilleur d'entre eux
            self.iterate_step()
            self.best_agent = sorted(self.all_agents, key=lambda a: a.value_at_pb)[0]

            if self.will_store:
                self.all_coords[self.step] = np.array([a.coords for a in self.all_agents])
            if self.known_solution is not None:  # vérification précision
                if LA.norm(self.best_agent.pb_coords - self.known_solution) < self.precision:
                    return (self.best_agent.pb_coords, self.best_agent.value_at_pb, "Precision")

        return (self.best_agent.pb_coords, self.best_agent.value_at_pb)

    def iterate_step(self):
        '''[effectue 1 etape de l'algorithme]'''
        for agent in self.all_agents:
            agent.update_pos(self.compute_velocity(agent))
            current_value = self.objective_function(agent.coords)
            self.current_cost += 1
            if current_value < agent.value_at_pb:
                agent.update_pb(current_value)

        self.inertia = np.maximum(self.inertia_prop[1], self.inertia - self.inertia_prop[2])

    def compute_velocity(self, agent):
        '''[Calcul de la vitesse d'une particule]
        Arguments:
            agent {[agent_PSO]} -- [l'agent dont on veut la vitesse]

        Returns:
            [un vecteur dimx1] -- [vecteur vitesse]
        '''
        social_best = self.get_social(agent)
        # Notons que les coefficients phi2 et U2 sont dans le calcul du social best
        return self.inertia * agent.velocity + self.phi1 * rng.uniform(0, 1) * (agent.pb_coords - agent.coords) + (social_best - agent.coords)

    def get_social(self, agent):
        '''[Permet de calculer le terme d'influence sociale]'''
        res = np.zeros((self.dim, 1))
        n = len(agent.neighbors)
        for nei in agent.neighbors:
            res += (self.phi2 / n) * rng.uniform(0, 1) * nei.pb_coords

        return res

    def get_storage(self):
        return self.all_coords[:self.step]

    def reset(self):
        self.inertia = self.inertia_prop[0]
        self.current_cost = self.nb_agents
        self.all_agents = self.create_agents()
        self.best_agent = sorted(self.all_agents, key=lambda a: a.value_at_pb)[0]
        self.step = 0
        if self.will_store:
            self.all_coords = np.zeros(((self.max_cost // self.nb_agents) + 1, self.nb_agents, self.dim, 1))
            self.all_coords[0] = np.array([a.coords for a in self.all_agents])


# Exemples de topologies
def SPSO_matrix(n):
    '''[renvoie la matrice signifiant que toutes les particules ont accès à toutes les autres, y-compris elles-mêmes]
    Ideal pour le fully-informed PSO'''
    return np.ones((n, n))

def UPSO_matrix(n):
    '''[comme la spso, mais sans les communications des particules avec elles-mêmes]'''
    return np.ones((n, n)) - np.eye(n)