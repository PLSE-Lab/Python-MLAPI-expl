#   This file is part of EAP.
#
#    EAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    EAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with EAP. If not, see <http://www.gnu.org/licenses/>.

# This version gives a public score of 0.79426

def mydeap(mungedtrain):
    
    import operator
    import math
    import random
    
    import numpy
    
    from deap import algorithms
    from deap import base
    from deap import creator
    from deap import tools
    from deap import gp
    
    inputs = mungedtrain.iloc[:,2:10].values.tolist()
    outputs = mungedtrain['Survived'].values.tolist()
    
    # Define new functions
    def protectedDiv(left, right):
        try:
            return left / right
        except ZeroDivisionError:
            return 1
    
    pset = gp.PrimitiveSet("MAIN", 8) # eight input
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protectedDiv, 2)
    pset.addPrimitive(operator.neg, 1)
    pset.addPrimitive(math.cos, 1)
    pset.addPrimitive(math.sin, 1)
    pset.addPrimitive(max, 2)
    pset.addPrimitive(min, 2) # add more?
    pset.addEphemeralConstant("rand101", lambda: random.uniform(-10,10)) # adjust?
    pset.renameArguments(ARG0='x1')
    pset.renameArguments(ARG1='x2')
    pset.renameArguments(ARG2='x3')
    pset.renameArguments(ARG3='x4')
    pset.renameArguments(ARG4='x5')
    pset.renameArguments(ARG5='x6')
    pset.renameArguments(ARG6='x7')
    pset.renameArguments(ARG7='x8')

    
    creator.create("FitnessMin", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3) #
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    
    def evalSymbReg(individual):
        # Transform the tree expression in a callable function
        func = toolbox.compile(expr=individual)
        # Evaluate the accuracy
        return sum(round(1.-(1./(1.+numpy.exp(-func(*in_))))) == out for in_, out in zip(inputs, outputs))/len(mungedtrain),
    
    toolbox.register("evaluate", evalSymbReg)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    
    
    
    
    random.seed(318)
    
    pop = toolbox.population(n=300) #
    hof = tools.HallOfFame(1)
    
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)
    
    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.2, 100, stats=mstats,
                                   halloffame=hof, verbose=True) #
    
    print(hof[0])
    func2 =toolbox.compile(expr=hof[0])
    return func2
    
import numpy as np
import pandas as pd

def Outputs(data):
    return np.round(1.-(1./(1.+np.exp(-data))))
    
def MungeData(data):
    # Sex
    data.drop(['Ticket', 'Name'], inplace=True, axis=1)
    data.Sex.fillna('0', inplace=True)
    data.loc[data.Sex != 'male', 'Sex'] = 0
    data.loc[data.Sex == 'male', 'Sex'] = 1
    # Cabin
    cabin_const = 1
    data.Cabin.fillna(str(cabin_const), inplace=True)
    data.loc[data.Cabin.str[0] == 'A', 'Cabin'] = cabin_const
    data.loc[data.Cabin.str[0] == 'B', 'Cabin'] = cabin_const
    data.loc[data.Cabin.str[0] == 'C', 'Cabin'] = cabin_const
    data.loc[data.Cabin.str[0] == 'D', 'Cabin'] = cabin_const
    data.loc[data.Cabin.str[0] == 'E', 'Cabin'] = cabin_const
    data.loc[data.Cabin.str[0] == 'F', 'Cabin'] = cabin_const
    data.loc[data.Cabin.str[0] == 'G', 'Cabin'] = cabin_const
    data.loc[data.Cabin.str[0] == 'T', 'Cabin'] = cabin_const
    # Embarked
    data.loc[data.Embarked == 'C', 'Embarked'] = 1
    data.loc[data.Embarked == 'Q', 'Embarked'] = 2
    data.loc[data.Embarked == 'S', 'Embarked'] = 3
    data.Embarked.fillna(0, inplace=True)
    data.fillna(-1, inplace=True)
    return data.astype(float)

    

if __name__ == "__main__":
    train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
    test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )
    mungedtrain = MungeData(train)
    
    #GP
    GeneticFunction = mydeap(mungedtrain)
    
    #test
    mytrain = mungedtrain.iloc[:,2:10].values.tolist()
    trainPredictions = Outputs(np.array([GeneticFunction(*x) for x in mytrain]))

    pdtrain = pd.DataFrame({'PassengerId': mungedtrain.PassengerId.astype(int),
                            'Predicted': trainPredictions.astype(int),
                            'Survived': mungedtrain.Survived.astype(int)})
    pdtrain.to_csv('MYgptrain.csv', index=False)
    from sklearn.metrics import accuracy_score
    print(accuracy_score(mungedtrain.Survived.astype(int),trainPredictions.astype(int)))
    
    mungedtest = MungeData(test)
    mytest = mungedtest.iloc[:,1:9].values.tolist()
    testPredictions = Outputs(np.array([GeneticFunction(*x) for x in mytest]))

    pdtest = pd.DataFrame({'PassengerId': mungedtest.PassengerId.astype(int),
                            'Survived': testPredictions.astype(int)})
    pdtest.to_csv('gptest_another.csv', index=False)
