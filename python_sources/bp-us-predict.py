import numpy as np
from pandas import read_csv, DataFrame, merge
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

class USPredictor:
    def __init__(self, layers, learning_rate, momentum, party, candidate, features, epoch, bias):
        self.input = layers[0]
        self.hidden = layers[1]
        self.output = layers[2]
        self.learning_rate = learning_rate;
        self.momentum = momentum
        self.party = party
        self.candidate = candidate
        self.features = features
        self.epoch = epoch
        self.bias = bias

        self.datas = "../input/"
        self.county_facts = self.datas + "county_facts.csv"
        self.county_facts_dic = self.datas + "county_facts_dictionary.csv"
        self.primary_result_file = self.datas + "primary_results.csv"

    def prepare_datas(self):
        self.__prepare_demographic()
        self.__prepare_primary_result()

    def prepare_datas_for_nn(self):
        self.__get_candidate_votes()

        self.dataset = SupervisedDataSet(len(self.features), 1)

        x = self.candidate_data_train[self.features]
        y = self.candidate_data_train.percent

        for i in range(0, x.shape[0]):
            t = (
                x.Population[i],
                x.AgeGreaterThan65[i],
                x.Black[i],
                x.Latino[i],
                x.White[i],
                x.HighSchool[i],
                x.Bachelors[i],
                x.MedianHousehold[i],
                x.LessThanPowertyLevel[i],
                x.females[i]
            )
            self.dataset.addSample(t, y[i])

    def build_train_network(self):
        self.neural_network = buildNetwork(self.input, self.hidden, self.output, bias=self.bias)
        trainer = BackpropTrainer(self.neural_network, self.dataset, learningrate=self.learning_rate, momentum=self.momentum)
        self.neural_network.reset()

        for x in range(1, self.epoch):
            error = trainer.train()
            if x % 1000 == 0:
                print("{}: {}".format(x, error), end='\n')


    def test_network(self):
        y_real_array = y_pred_array = None
        datas = {}
        states = []
        for state in self.candidate_data_test.state:
            states.append(state)
            x_test = self.__test_data(state)
            y_test = self.candidate_data_test[(self.candidate_data_test.state == state)].percent

            for index, value in y_test.iteritems():
                y_test = value

            y_pred = self.neural_network.activate(x_test)
            if y_pred_array == None:
                y_pred_array = np.array(y_pred)
                y_real_array = np.array(y_test)
            else:
                y_pred_array = np.append(y_pred_array, y_pred)
                y_real_array = np.append(y_real_array, y_test)
            d = {"State": state, "Real": y_test, "Predicted": y_pred}
            d = DataFrame(d)
            print(d)


        error = mean_squared_error(y_real_array, y_pred_array)
        print("MSE: ", error)
        return error, states, y_pred_array, y_real_array



    def predict(self, states):
        print("\n", end="")
        print("Prediction future result of {} ".format(self.candidate))
        y_pred_array = None
        for state in states:
            x_test = self.__test_data(state)
            y_pred = self.neural_network.activate(x_test)
            if y_pred_array == None:
                y_pred_array = np.array(y_pred)
            else:
                y_pred_array = np.append(y_pred_array, y_pred)
            d = {"State": state, "Predicted": y_pred}
            print(DataFrame(d))
        return y_pred_array


    def __test_data(self, state):
        demog_test = self.demog[(self.demog.area_name == state)]
        x_test = demog_test[self.features].reset_index().drop('index', axis=1)
        x_test = (
            x_test.Population[0],
            x_test.AgeGreaterThan65[0],
            x_test.Black[0],
            x_test.Latino[0],
            x_test.White[0],
            x_test.HighSchool[0],
            x_test.Bachelors[0],
            x_test.MedianHousehold[0],
            x_test.LessThanPowertyLevel[0],
            x_test.females[0]
        )
        return x_test

    def __get_candidate_votes(self):
        self.filtered_primary_result = self.primary_result[(self.primary_result.party == self.party)]

        self.votes_by_state = [[candidate, state, party]
                               for candidate in self.filtered_primary_result.candidate.unique()
                               for state in self.filtered_primary_result.state.unique()
                               for party in self.filtered_primary_result.party.unique()]

        for ind in self.votes_by_state:
            ind.append(self.filtered_primary_result[(self.filtered_primary_result.candidate == ind[0])
                                                    & (self.filtered_primary_result.state == ind[1])].votes.sum())
            ind.append(ind[3] / self.filtered_primary_result[self.filtered_primary_result.state == ind[1]].votes.sum())


        self.votes_by_state = DataFrame(self.votes_by_state,
                                        columns=['candidate', 'state', 'party', 'votes', 'percent'])

        self.filtered_data = merge(self.votes_by_state, self.demog, how="inner", left_on='state', right_on='area_name')
        self.filtered_data.drop('state_abbreviation', axis=1, inplace=True)

        self.candidate_data = self.filtered_data[(self.filtered_data.candidate == self.candidate)]
        self.candidate_data = self.candidate_data.reset_index()

        self.candidate_data.drop('index', axis=1, inplace=True)

        self.candidate_data_train, self.candidate_data_test = train_test_split(self.candidate_data, test_size=0.20)

        self.candidate_data_train = self.candidate_data_train.reset_index().drop('index', axis=1)
        self.candidate_data_test = self.candidate_data_test.reset_index().drop('index', axis=1)


        print(self.candidate_data_train)

    def __prepare_primary_result(self):
        self.primary_result = read_csv(self.primary_result_file)
        self.primary_result = self.primary_result[
            (self.primary_result.candidate != ' Uncommitted') & (self.primary_result.candidate != 'No Preference')]

    def __prepare_demographic(self):
        self.demog = read_csv(self.county_facts)
        self.demog = self.demog[
            ['fips',
             'area_name',
             'state_abbreviation',
             'PST045214',
             'AGE775214',
             'RHI225214',
             'RHI725214',
             'RHI825214',
             'EDU635213',
             'EDU685213',
             'INC110213',
             'PVY020213',
             'SEX255214']
        ]

        self.demog.rename(columns=
        {
            'PST045214': self.features[0],
            'AGE775214': self.features[1],
            'RHI225214': self.features[2],
            'RHI725214': self.features[3],
            'RHI825214': self.features[4],
            'EDU635213': self.features[5],
            'EDU685213': self.features[6],
            'INC110213': self.features[7],
            'PVY020213': self.features[8],
            'SEX255214': self.features[9]
        },
            inplace=True)
        for column in self.demog:
            if (column != "fips" and column != "area_name" and column != "state_abbreviation"):
                self.demog[column] = self.demog[column].divide(self.demog[column].max())
                # self.demog[column] = self.demog[column].subtract(self.demog[column].mean()).divide(self.demog[column].std())



def main():
    features = ['Population',
                'AgeGreaterThan65',
                'Black',
                'Latino',
                'White',
                'HighSchool',
                'Bachelors',
                'MedianHousehold',
                'LessThanPowertyLevel',
                'females']
    candidate = "Hillary Clinton"

    for i in range(0, 5):
        us_predictor = USPredictor([len(features), 3, 1], 0.1, 0.3, "Democrat", candidate, features, 5000, True)
        us_predictor.prepare_datas()
        us_predictor.prepare_datas_for_nn()
        us_predictor.build_train_network()
        error, states, y_pred_array, y_real_array = us_predictor.test_network()
        y_pred_future_array = us_predictor.predict(["Alaska", "Massachusetts", "New York", "Minnesota"])
        y_pred_future_array2 = us_predictor.predict(["Montana", "North Dakota", "South Dakota", "California", "New Mexico"])

        # Specifically for Hillary clinton
        y_real_future_array = [0.184, 0.501, 0.580, 0.384]

        # # Specifically for Berinie Sanders
        # y_real_future_array = [0.816, 0.487 , 0.420, 0.616]

        y_real_future_array2 = [0, 0, 0, 0, 0]

        data = {
            candidate: states,
            'Predicted': y_pred_array,
            'Real': y_real_array
        }
        data2 = {
            candidate: ["Alaska", "Massachusetts", "New York", "Minnesota"],
            'Predicted': y_pred_future_array,
            'Real': y_real_future_array
        }
        data3 = {
            candidate: ["Montana", "North Dakota", "South Dakota", "California", "New Mexico"],
            'Predicted': y_pred_future_array2,
            'Real': y_real_future_array2
        }

        df = DataFrame(data, columns = [candidate, 'Predicted', 'Real'])
        df2 = DataFrame(data2, columns = [candidate, 'Predicted', 'Real'])
        df3 = DataFrame(data3, columns = [candidate, 'Predicted', 'Real'])

        # figure = plt.figure(figsize=(10,5))
        figure = plt.figure()

        ax = figure.add_subplot(3, 1, 1)
        plot(df, candidate, ax, '{} Predicted votes in epoch #{}'.format(candidate, i + 1))

        ax = figure.add_subplot(3, 1, 2)
        plot(df2, candidate, ax, '{} Predicted votes in epoch #{}'.format(candidate, i + 1))

        ax = figure.add_subplot(3, 1, 3)
        plot(df3, candidate, ax, '{} Predicted votes in epoch #{}'.format(candidate, i + 1))

    plt.show()



def plot(df, candidate, ax, title):
    # Setting the positions and width for the bars
    pos = list(range(len(df['Predicted'])))
    width = 0.25

    # Plotting the bars


    # Create a bar with Predicted data,
    # in position pos,
    plt.bar(pos,
            #using df['Predicted'] data,
            df['Predicted'],
            # of width
            width,
            # with alpha 0.5
            alpha=0.5,
            # with color
            color='#EE3224',
            # with label the first value in first_name
            label=df[candidate][0])

    # Create a bar with Real data,
    # in position pos + some width buffer,
    plt.bar([p + width for p in pos],
            #using df['Real'] data,
            df['Real'],
            # of width
            width,
            # with alpha 0.5
            alpha=0.5,
            # with color
            color='#F78F1E',
            # with label the second value in first_name
            label=df[candidate][1])

    # Set the y axis label
    ax.set_ylabel('Score')

    # Set the chart's title
    ax.set_title(title)

    # Set the position of the x ticks
    ax.set_xticks([p + 1.5 * width for p in pos])

    # Set the labels for the x ticks
    ax.set_xticklabels(df[candidate])

    # Setting the x-axis and y-axis limits
    plt.xlim(min(pos) - width, max(pos)+ width*4)
    plt.ylim([0, 1] )

    # Adding the legend and showing the plot
    plt.legend(['Predicted', 'Real'], loc='upper left')
    plt.grid()


if __name__ == '__main__':
    main()
