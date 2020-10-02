# coding: utf-8

import sys
import os
import time
from IPython.display import HTML, display
from IPython.display import Image
import pandas as pd
import numpy as np
import random
from decimal import Decimal
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
maxInt = sys.maxsize

############################################################################################################################################################################################################
# README
############################################################################################################################################################################################################

def readme():

    print("--- Author ---\n")
    print(
        "This package was created by Rafael Klanfer Nunes please refer:\n"
        "Code title: Benford's Law approach to detect data manipulation\n"
        "Date: 03/april/2020\n"
        "Code Version: 1.0\n"
        "Type: Python 3\n"
        "URL: https://www.linkedin.com/in/rafaelknunes\n"
    )

    print("--- Important Functions ---\n")
    print("1) hint_lessons(data): Use this function to learn what is the best sample size for your analysis.\n"
          "2) benford(args): The main function. Below learn more about its args.\n")

    print("--- KEY WORDS ---\n")
    print(
          "1) DIGIT or positional digit (D1, D2, D3): Refers to the position of the value we are looking for. Benfod's Law has an specific distribution for the first, second and third digit of a value. For example, consider the ammount of refund some employee asked in some period: $54.234. In this case the first digit is 5, the second is 4 and the third is 2.\n"
          "2) VALUE: In order to avoid confusion, we will call the number that we want to analyse as the **VALUE**. For example, if the user wants to analyse fraud or manipulation on data regarding employee's reimbursements, or daily revenues, or any other quantities, these will be called VALUES.\n"
          "3) NUMBER (N0, N1, .., N9): In order to avoid confusion, lets call NUMBER what we find in a certain positional digit of a value. For example, suppose we have the value 54.234 and we are analysing the digit 1 (D1). So, the number we get is 5 (N5). For digit 2 (D2) we get the number 4 (N4) and for the digit 3 (D3) we get the number 2 (N2).\n"
          "4) UNIT: Suppose again that we are analysing a dataset from employees reimbursements. In this case, a unit is an specific employee. This is important since the user may choose to produce results for every employee. In any case, the code will always produce results for the entire dataset, considering all units as if they were one.\n"
          "5) LESSONS: Lets call every row of the dataset a lesson. For example, a reimbursement value of some employee is a lesson.\n"
          )

    print("--- Main function: benford(Data, type_analysis, chi_sq_rounds, lessons_D1, lessons_D2, lessons_D3, graphs_worst, graphs_best) ---\n")
    print(
        "1) Data: This code MUST receive a dataset with only two columns. One must be of numeric type and should contain the values to be analyzed. (Total number of population, value of receipts, value of expenses, etc.). The second column must be of text type, and should contain the name of the units that are being analyzed. (cities, companies, employees, etc.). That way, each row represents a lesson. It shows the value associated to a unit in a certain period of time (day, week, month, etc.)\n"
        "2) Type of analysis: 1 if the analysis ONLY focus on the entire database as a whole. Anything else if the analysis run over each grouped unit in the database.\n"
        "3) Rounds for chi-squared test: How many times this code will run in order to generate a chi-squared value. This value is the simple mean from these several runs.\n"
        "4) Sample size when analysing the first digit. Important: Units that dont match the asked sample size will be disregarded.\n"
        "5) Sample size when analysing the second digit. Important: Units that dont match the asked sample size will be disregarded.\n"
        "6) Sample size when analysing the third digit. Important: Units that dont match the asked sample size will be disregarded.\n"
        "7) Number of graphs to save from units with WORST chi-squared scores.\n"
        "8) Number of graphs to save from units with BEST chi-squared scores.\n"
        "9) xlsx_path: path to save the excel file with the results. For example type 'output/final_result.xslx' - Always use .xlsx in the end.\n"
        "10) graph_path: Path for the folder where we want to save the graphs. For example type: 'output/endfolder' - Never type '/' in the end.\n"
    )

    print("--- OUTPUTS ---\n")
    print(
        "1) Excel file containing 3 sheets, each referring to the digit analyzed (D1, D2, D3). Each sheet contains:\n"
        "- The observed frequency grouped by unit, plus an aggregated analysis, considering all units as one. User may choose to run only the aggregated analysis. \n"
        "- Theoretical frequency by Benford's Law for the corresponding digit.\n"
        "- The aggregated frequency observed. This is the result if we consider all units as if they were one. In this case, the program uses a sample from the entire dataset to make its analysis.\n"
        "- The value of the Pearson's chi-squared test for each group.\n"
        "- An additional column with the chi-squared value for each group, but in this case the value is the simple average of several rounds of calculations. Rounds defined by user.\n"
        "2) The code also return a list with these 3 tables saved on excel file.\n"
        "3) The code also produces several graphs comparing the frequency observed in units that showed high or low chi-squared values. Defined by user.\n"
    )
    return print("-----------------------  Type: hints(Data, type_analysis) before running the main code!  -----------------------\n\n")

############################################################################################################################################################################################################
# HINTS
############################################################################################################################################################################################################

def hints(df, type_analysis):
    '''
    :param df: DataFrame with the observed values. Must have one column named "unit" (object) and other column named "value" (numeric).
    :param type_analysis: 1 if the analysis ONLY focus on the entire database as a whole. Anything else if the analysis run over each grouped unit in the database.
    '''
    
        # Chi-squared-table
    critical_values = [{'first digit': 20.09, 'second digit':21.67, 'third digit':21.67}, {'first digit': 15.51, 'second digit':16.92, 'third digit':16.92}, {'first digit': 13.36, 'second digit':14.68, 'third digit':14.68} ]
    chi_sq_table = pd.DataFrame(critical_values, index =['1% of significance', '5% of significance', '10% of significance']) 
    print("----------------------------------------\n# Chi-squared table of critical values #\n----------------------------------------\n\n", chi_sq_table)
    print("\n\n")

    # Adjusting dataFrame types
    if (df.dtypes[0] == "object" and df.dtypes[1] != "object"):
        df.columns = ['unit', 'value']
        df['value'].astype(float)
    elif (df.dtypes[0] != "object" and df.dtypes[1] == "object"):
        df.columns = ['value', 'unit']
        df['value'].astype(float)
    else:
        print(
            "WARNING! Are you using Decimal??The dataset must have one column with type TEXT named 'unit' and the other should be named 'value' and be of type NUMBER.\n")

    # TYPE OF ANALYSIS
    if(type_analysis == 1):
        df["unit"] = 'to_be_deleted'
    else:
        pass

    ###################################################
    # LESSONS
    ###################################################

    # First verify the max number of lessons the units have, considering that for D1 we will use values bigger than 1, for D2, bigger than 10 and for D3, bigger than 100.
    count_big = [0,0,0]
    # Number of lessons with value greater than 1. For the unit with most lessons.
    df_aux1 = (df.loc[df['value'] >= 1]).copy()
    df_aux1['unit_count'] = df_aux1.groupby(['unit'])['value'].transform('count')
    count_max = df_aux1['unit_count'].max()
    count_big[0] = count_max
    # Number of lessons with value greater than 10. For the unit with most lessons.
    df_aux10 = (df.loc[df['value'] >= 10]).copy()
    df_aux10['unit_count'] = df_aux10.groupby(['unit'])['value'].transform('count')
    count_max = df_aux10['unit_count'].max()
    count_big[1] = count_max
    # Number of lessons with value greater than 100. For the unit with most lessons.
    df_aux100 = (df.loc[df['value'] >= 100]).copy()
    df_aux100['unit_count'] = df_aux100.groupby(['unit'])['value'].transform('count')
    count_max = df_aux100['unit_count'].max()
    count_big[2] = count_max

    if (type_analysis == 1):
        print(f"This is an aggregated analysis. The max sample size for D1 is: {count_big[0]}, for D2: {count_big[1]} and for D3 is: {count_big[2]}.\n")
    else:
        print(f'Sample size when analysing the first digit. Consider that the unit with more lessons has only {count_big[0]} lessons bigger than 1.\n')
        print(f'Sample size when analysing the second digit. Consider that the unit with more lessons has only {count_big[1]} lessons bigger than 1.\n')
        print(f'Sample size when analysing the third digit. Consider that the unit with more lessons has only {count_big[2]} lessons bigger than 1.\n')

    return print("--- End Hint ---\n\n")

############################################################################################################################################################################################################
# FUNCTIONS: benford(args)
############################################################################################################################################################################################################

# GLOBAL VARIABLES
# This variables will store the final result for observed frequencies in each of the 3 digits. Then, we can show results without needing to open the resulted spreadsheet.
results_D1 = pd.DataFrame()
results_D2 = pd.DataFrame()
results_D3 = pd.DataFrame()

def find_number(value, positional_digit):
    '''
    :param num: The value being analyzed. (Quantities, expenses, etc.)
    :param positional_digit: The digit, or position of the value being analyzed. May be the first, second or third position of a value. (D1, D2, D3)
    return: From a given value and a positional digit (D1, D2, D3) returns the observed number (0-9). For example: find_number(3284, 3) = 8.
    '''
    
    # We must avoid the code to extract the "dot" of a number.
    value = int(value)
    number = str(value)
    if(positional_digit == 1):
        number = number[0]
        number = int(number)
    elif (positional_digit == 2):
        number = number[1]
        number = int(number)
    elif(positional_digit == 3):
        number = number[2]
        number = int(number)
    else:
        print("\n WARNING! Invalid Digit \n")

    return number

############################################################################################################################################################################################################

def get_theoretical_benford(positional_digit):
    '''
    :param positional_digit: The digit, or position of the value being analyzed. May be the first, second or third position of a value. (D1, D2, D3)
    return: A list with the theoretical frequencies predicted by Benford's Law for an specific positional digit.
    '''
    if(positional_digit == 1):
        list = {"units": f"_Benford_D{positional_digit}", "N0": 0, "N1": 0.301, "N2": 0.176, "N3": 0.125, "N4": 0.097, "N5": 0.079, "N6": 0.067, "N7": 0.058, "N8": 0.051, "N9": 0.046, "lessons": 1}
    elif(positional_digit == 2):
        list = {"units": f"_Benford_D{positional_digit}", "N0": 0.12, "N1": 0.114, "N2": 0.109, "N3": 0.104, "N4": 0.100, "N5": 0.097, "N6": 0.093, "N7": 0.090, "N8": 0.088, "N9": 0.085, "lessons": 1}
    elif(positional_digit == 3):
        list = {"units": f"_Benford_D{positional_digit}", "N0": 0.102, "N1": 0.101, "N2": 0.101, "N3": 0.101, "N4": 0.100, "N5": 0.100, "N6": 0.099, "N7": 0.099, "N8": 0.099, "N9": 0.098, "lessons": 1}
    return list

############################################################################################################################################################################################################

def create_graph(df_grouped, positional_digit, unit, chi_sq, lessons, chi_sq_rounds, graph_path):
    '''
    :param df_grouped: DataFrame with the observed frequencies grouped by unit.
    :param positional_digit: The digit, or position of the value being analyzed. May be the first, second or third position of a value. (D1, D2, D3)
    :param unit: Name of the unit which graph will be plotted.
    :param chi_sq: Value of the calculated chi-squared for this unit and this positional digit.
    :param lessons: How many lessons were considered in this analysis.
    :param chi_sq_rounds: How many times this code ran in order to generate a chi-squared value. This value is the simple mean from the several runs.
    :param graph_path: Path for the folder where we want to save the graphs.
    Important: We must always consider the same number of lessons for each unit evaluated.
    return: A graph containing the frequency observed for each number (0-9) in the positional digit and unit analyzed.
    '''

    # 1) Creates a new dataFrame with the columns we want.
    new_df = df_grouped[['N0', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8', 'N9']].copy()

    # 2) Create the X axis
    eixoX = ['N0', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8', 'N9']

    # 3) Create both Y axis. One for the theoretical benford frequency, and another for the observed frequency.
    eixoY_observado_inicial = list(new_df.loc[unit, :])
    eixoY_teorico_inicial = list(new_df.loc[f"_Benford_D{positional_digit}", :])
    # 3.1) Since we are working with frequencies, lets multiply by 100 to show percentages in the graph.
    eixoY_observado_100 = np.array(eixoY_observado_inicial)*100
    eixoY_teorico_100 = np.array(eixoY_teorico_inicial)*100
    # 3.2) Rounding 1 place after the decimal point.
    eixoY_observado = [round(elem, 1) for elem in eixoY_observado_100]
    eixoY_teorico = [round(elem, 1) for elem in eixoY_teorico_100]

    # 4) Definitions
    # 4.1) Max value among all values. It defines the graph's height.
    max_total = max(max(eixoY_teorico), max(eixoY_observado))
    # 4.2) Size of X axis labels.
    centroX = np.arange(len(eixoX))
    # 4.3) Size of Y axis labels.
    centroY = np.arange(0, 1.3*max_total, 1.3*max_total/10)
    # 4.4) Bar width.
    bar_width = 0.25

    # 5) Start Graph.
    fig = plt.figure(num=None, figsize=(8, 8), dpi=120, facecolor='w', edgecolor='k')
    ax = fig.add_subplot()
    # 5.1) Start Bars.
    bar_eixoY_observado = ax.bar(centroX - bar_width / 1.5, eixoY_observado, bar_width, label='Observed Freq.', color='tan')
    bar_eixoY_teorico = ax.bar(centroX + bar_width / 1.5, eixoY_teorico, bar_width, label=f'Theoretical Freq. for D{positional_digit}', color='pink')
    # 5.2) Number of X axis centers.
    ax.set_xticks(centroX)
    # 5.3) Name of X axis centers.
    ax.set_xticklabels(eixoX)
    # 5.4) Number of y axis centers.
    ax.set_yticks(centroY)
    # 5.5) Name of Y axis centers. (empty)
    ax.set_yticklabels("")
    # 5.6) Graph title
    ax.set_title(f"Benford's Law for D{positional_digit} \n Unit: {unit} || Lessons: {lessons}"
                 f"\n Average Chi-squared value after {chi_sq_rounds} rounds: {chi_sq}")

    # 6) Labels
    # Defining labels
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=5)

    # 7.1) Create bar label for Y (bar_eixoY_teorico)
    for rect in bar_eixoY_teorico:
        height = rect.get_height()
        ax.annotate('{}%'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 2),
                    textcoords="offset points",size=7, color="black",
                    ha='center', va='baseline')

    # 7.2) Create bar label for Y (bar_eixoY_observado)
    for rect in bar_eixoY_observado:
        height = rect.get_height()
        ax.annotate('{}%'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(-4, 4),
                    textcoords="offset points",size=7, color="black",
                    ha='center', va='baseline')

    # 8) Final
    # Adjust of layout
    fig.tight_layout()
    #plt.show()
    # Save graph
    plt.savefig(f"{graph_path}/D{positional_digit}_{unit}.png")
    # Close figure
    plt.close(fig)

    return print("Graph generated with success!")

############################################################################################################################################################################################################

def get_chi_squared(df_grouped, positional_digit):
    '''
    :param df_grouped: DataFrame with the observed frequencies grouped by unit.
    This dataFrame has the final frequencies for each unit. So it is a treated dataFrame.
    :param positional_digit: The digit, or position of the value being analyzed. May be the first, second or third position of a value. (D1, D2, D3)
    return: Series with observed values of chi_sq for each unit. Places in the last column.
    '''

    ult_linha = df_grouped.shape[0] - 1
    ult_coluna = df_grouped.shape[1] - 1

    # For each df row (observed frequencies of a unit) we sum the squared difference between the observed and theoretical values for the positional digit.
    # The values of the theoretical frequencies are in the last row.
    for lin in range(ult_linha + 1):
        # When analysing the D1, we dont need to care about the number 0.
        # For this reason we start on the third column, since the first column contains the number of lessons, and the second the analysis for N0.
        if(positional_digit == 1):
            for col in range(2, ult_coluna, 1):
                # Set the chi_sq column with the value of the latter chi_sq PLUS the difference of obs. freq. and theoretical freq. to square.
                df_grouped.iloc[lin, ult_coluna] = df_grouped.iloc[lin, ult_coluna] + ( ( (df_grouped.iloc[lin, col] - df_grouped.iloc[ult_linha, col])**2 ) / df_grouped.iloc[ult_linha, col] )
        # Here we consider the analysis of the number 0, which is in the second column.
        else:
            for col in range(1, ult_coluna, 1):
                # Set the chi_sq column with the value of the latter chi_sq PLUS the difference of obs. freq. and theoretical freq. to square.
                df_grouped.iloc[lin, ult_coluna] = df_grouped.iloc[lin, ult_coluna] + ( ( (df_grouped.iloc[lin, col] - df_grouped.iloc[ult_linha, col])**2 ) / df_grouped.iloc[ult_linha, col] )
        # After that multiply the partial chi_sq by the total number of lessons, since we are working with relatives frequencies.
        df_grouped.iloc[lin, ult_coluna] = df_grouped.iloc[lin, ult_coluna] * df_grouped.iloc[lin, 0]

    return df_grouped

############################################################################################################################################################################################################

def get_observed_frequency(df, positional_digit):
    '''
    :param df: DataFrame with the observed values. Must have one column named "unit" (object) and other column named "value" (numeric).
    :param positional_digit: The digit, or position of the value being analyzed. May be the first, second or third position of a value. (D1, D2, D3)
    return: A dataFrame with the observed frequencies of each number (0-9) in the positional digit analyzed. Grouped by unit.
    '''

    ###################################################
    # VARIABLE
    ###################################################

    # Count the number of unique units we have.
    num_units = df['unit'].nunique(dropna=True)
    # Matrix where each row corresponds to a single unit and the columns will receive the relative frequency observed for the numbers 0-9 in the analyzed positional digit.
    matriz_resultados = np.zeros((num_units, 11))
    # Vector that carries the results considering all units as if it was just one. An aggregated analysis.
    vetor_resultados_agregados = np.zeros((11))
    # Vector with the name of the units. This is a Series type since we want a column with index and values.
    vetor_units = pd.Series(df['unit'].unique())
    # Indicates which row (unit) we are working.
    linha_unit = 0
    # Indicates which column (N0-N9) we are working.
    indice = 0

    ###################################################
    # MAIN
    ###################################################

    # IMPORTANT. Always order by unit, since the code will orient itself by the change in unit names when setting the observed frequencies.
    df.sort_values("unit", inplace=True)
    # IMPORTANT: Reset index. Index should go from 0 to ..., since we wil use it for row indexing.
    df.reset_index(inplace=True, drop=True)
    # Start unit
    unit_anterior = df.loc[0, "unit"]

    # Runs for each row in dataFrame.
    # row[0] = index of the row | row[1] = unit | row[2] = value
    for row in range(df.shape[0]):
        # Actual unit. It changes as we change rows.
        unit_atual = df.loc[row, "unit"]
        # The code runs the same way while we are in the same unit lesson. When we change the unit in the original dataFrame, we must assign a new row to the result matrix.
        if (unit_atual == unit_anterior):
            # Assign to "indice" the number we found in the value's positional digit. It will be used to carry the counting on the right column of the result matrix. N0-N9.
            # Here we set "indice" with the value of the first positional digit D1.
            if(positional_digit == 1):
                indice = find_number(df.loc[row, "value"], positional_digit)
            # Here we set "indice" with the value of the second positional digit D2.
            elif(positional_digit == 2):
                indice = find_number(df.loc[row, "value"], positional_digit)
            # Here we set "indice" with the value of the third positional digit D3.
            elif(positional_digit == 3):
                indice = find_number(df.loc[row, "value"], positional_digit)
            else:
                print("\n WARNING! Invalid Digit \n")

            # matriz_resultados: Has the absolute observed frequency that each number (0-9) appears for each unit.
            # Now lets add 1 to the correct column obtained earlier. For example, if the analyzed lesson had value = 3423 then we add +1 to the column reffering to N3.
            matriz_resultados[linha_unit][indice] = matriz_resultados[linha_unit][indice] + 1
            # The 10th column counts the number of lessons analyzed so far. It will be useful to generate relative frequencies.
            matriz_resultados[linha_unit][10] = matriz_resultados[linha_unit][10] + 1

            # vetor_resultados_agregados: Has the absolute observed frequency that each number (0-9) appears for all the units if they were considered as one.
            vetor_resultados_agregados[indice] = vetor_resultados_agregados[indice] + 1
            # The 10th column counts the number of lessons analyzed so far. It will be useful to generate relative frequencies.
            vetor_resultados_agregados[10] = vetor_resultados_agregados[10] + 1

        # ELSE: Só ocorre quando há mudança de unit ao longo das linhas. Por isso a importancia da base estar ORDENADA!
        # Com ele, garantimos que a frequência observada passe a ser jogada na próxima linha da matriz que carrega os resultados.
        else:
            # Here we will change the row (unit) to assign the values.
            linha_unit = linha_unit + 1
            # Assign to "indice" the number we found in the value's positional digit. It will be used to carry the counting on the right column of the result matrix. N0-N9.
            # Here we set "indice" with the value of the first positional digit D1.
            if(positional_digit == 1):
                indice = find_number(df.loc[row, "value"], positional_digit)
            # Here we set "indice" with the value of the second positional digit D2.
            elif(positional_digit == 2):
                indice = find_number(df.loc[row, "value"], positional_digit)
            # Here we set "indice" with the value of the third positional digit D3.
            elif(positional_digit == 3):
                indice = find_number(df.loc[row, "value"], positional_digit)
            else:
                print("\n WARNING! Invalid Digit \n")

            # matriz_resultados: Has the absolute observed frequency that each number (0-9) appears for each unit.
            # Now lets add 1 to the correct column obtained earlier. For example, if the analyzed lesson had value = 3423 then we add +1 to the column reffering to N3.
            matriz_resultados[linha_unit][indice] = matriz_resultados[linha_unit][indice] + 1
            # The 10th column counts the number of lessons analyzed so far. It will be useful to generate relative frequencies.
            matriz_resultados[linha_unit][10] = matriz_resultados[linha_unit][10] + 1

            # vetor_resultados_agregados: Has the absolute observed frequency that each number (0-9) appears for all the units if they were considered as one.
            vetor_resultados_agregados[indice] = vetor_resultados_agregados[indice] + 1
            # The 10th column counts the number of lessons analyzed so far. It will be useful to generate relative frequencies.
            vetor_resultados_agregados[10] = vetor_resultados_agregados[10] + 1

        # Now lets say that the passed unit is the actual unit. This will be important to verify if after changing rows we changed units too.
        unit_anterior = unit_atual


    ###################################################
    # RESULTS
    ###################################################

    # 1) Create a grouped dataFrame
    # * rows: correspond to an specific unit. | * columns: name of the unit; number of lessons, observed relative frequency for the N0 to N9.
    # See that in order to produce relative frequencies, we dived the absolute frequency by the number of lessons used to calculate that frequency.
    df_grouped = pd.DataFrame({
        'units': vetor_units,
        'lessons': matriz_resultados[:, 10],
        'N0': matriz_resultados[:, 0] / matriz_resultados[:, 10],
        'N1': matriz_resultados[:, 1] / matriz_resultados[:, 10],
        'N2': matriz_resultados[:, 2] / matriz_resultados[:, 10],
        'N3': matriz_resultados[:, 3] / matriz_resultados[:, 10],
        'N4': matriz_resultados[:, 4] / matriz_resultados[:, 10],
        'N5': matriz_resultados[:, 5] / matriz_resultados[:, 10],
        'N6': matriz_resultados[:, 6] / matriz_resultados[:, 10],
        'N7': matriz_resultados[:, 7] / matriz_resultados[:, 10],
        'N8': matriz_resultados[:, 8] / matriz_resultados[:, 10],
        'N9': matriz_resultados[:, 9] / matriz_resultados[:, 10]
    })

    # 2) Add a row for the result, considering all units as if they were a single one.
    units_as_one = { "units": '_Aggregated', "lessons": vetor_resultados_agregados[10], "N0": vetor_resultados_agregados[0] / vetor_resultados_agregados[10],
                                "N1": vetor_resultados_agregados[1] / vetor_resultados_agregados[10], "N2": vetor_resultados_agregados[2] / vetor_resultados_agregados[10],
                                "N3": vetor_resultados_agregados[3] / vetor_resultados_agregados[10], "N4": vetor_resultados_agregados[4] / vetor_resultados_agregados[10],
                                "N5": vetor_resultados_agregados[5] / vetor_resultados_agregados[10], "N6": vetor_resultados_agregados[6] / vetor_resultados_agregados[10],
                                "N7": vetor_resultados_agregados[7] / vetor_resultados_agregados[10], "N8": vetor_resultados_agregados[8] / vetor_resultados_agregados[10],
                                "N9": vetor_resultados_agregados[9] / vetor_resultados_agregados[10]}
    # Append this row to the df_grouped
    df_grouped_v1 = df_grouped.append(units_as_one, ignore_index=True)

    # 3) Append to the dataFrame a row with theoretical Benford's Law frequencies, in the analyzed digit.
    df_grouped_v2 = df_grouped_v1.append(get_theoretical_benford(positional_digit), ignore_index=True)

    # 4) Assign unit names as index.
    df_grouped_v3 = df_grouped_v2.set_index(['units'])

    # 5) Add a column that will store chi-squared results.
    df_grouped_v3["chi_sq"] = 0

    # 5) Append the chi-squared results to the final dataFrame.
    df_grouped_final = get_chi_squared(df_grouped_v3, positional_digit)

    return df_grouped_final

############################################################################################################################################################################################################

def create_results(df, num_lessons, positional_digit, chi_sq_rounds, writer, num_graphs, type_analysis, graph_path):
    '''
    :param df: DataFrame with the observed values. Must have one column named "unit" (object) and other column named "value" (numeric).
    :param num_lessons: Minimum number of lessons per unit considering D1, D2, D3. If some unit has more than the min so we select a sample with the min. This depends on the positional digit.
    :param positional_digit: The digit, or position of the value being analyzed. May be the first, second or third position of a value. (D1, D2, D3)
    :param chi_sq_rounds: How many times this code ran in order to generate a chi-squared value. This value is the simple mean from the several runs.
    :param writer: Path where we are exporting the results. Defined out of the function, since we will reuse it to save results for D1, D2, D3 in same spreadsheet.
    :param num_graphs: How many graphs to be created. num_graphs[0] = number of graphs with the highest chi_sq. num_graphs[1] = number of graphs with worst chi_sq.
    :param type_analysis: 1 if the analysis ONLY focus on the entire database as a whole. Anything else if the analysis run over each grouped unit in the database.
    :param graph_path: Path for the folder where we want to save the graphs.
    return: A spreadsheet with obs. rel. freq. for each number 0-9 on the actual positional digit, and its respective chi_sq plus the Theoretical Benford numbers.
    For every unit and one row considering all units as if they were one.
    '''
    
    df_copy = df.copy()

    ###################################################
    # PREPARATION
    ###################################################

    # ITEM 1: When analysing D1, consider values greater than 1. For D2, greater than 10. For D3, greater than 100.
    if(positional_digit == 1):
        df_inicial_ajustado = (df_copy.loc[df_copy['value'] >= 1]).copy()
    elif(positional_digit == 2):
        df_inicial_ajustado = (df_copy.loc[df_copy['value'] >= 10]).copy()
    elif(positional_digit == 3):
        df_inicial_ajustado = (df_copy.loc[df_copy['value'] >= 100]).copy()
    else:
        print("\n WARNING! Invalid Digit \n")

    # ITEM 2: Exclude units where the number of lessons are smaller than the minimum.
    # Step 1: Create an addition column to store the number of lessons.
    df_inicial_ajustado['unit_count'] = df_inicial_ajustado.groupby(['unit'])['value'].transform('count')

    # Step 2: Keep only units which lesson count is bigger than the minimum.
    # First check if the unit with the biggest number of lessons still has enough lessons. Otherwise the dataFrame would be empty.
    unit_count_max = df_inicial_ajustado['unit_count'].max()
    if(unit_count_max < num_lessons):
        print(f"WARNING!: The user wants this number of lessons: {num_lessons}. However, the unit with more information has only: {unit_count_max} lessons")
        print(f"Positional Digit: {positional_digit}")
        sys.exit()
    else:
        df_inicial_maior_que_minimo = (df_inicial_ajustado.loc[df_inicial_ajustado['unit_count'] >= num_lessons]).copy()

    # ITEM 3: Now some units may have more than the minimum number of lessons. Lets make everyone equal.
    # replace=False: Sampling without substition. | random_state: seed value. | group_keys=False: does not create group keys.
    df_standard = (df_inicial_maior_que_minimo.groupby('unit', group_keys=False).apply(pd.DataFrame.sample, n=num_lessons, replace=False, random_state=1)).copy()

    ####################################################
    # MAIN
    ####################################################

    # ITEM 1: Create the dataFrame with the obs. freq. for each digit. The last column will show chi_sq results.
    df_final = get_observed_frequency(df_standard, positional_digit)
    # Store the chi_sq values in an Series.
    df_chi_sq = df_final.iloc[:, df_final.shape[1] - 1]

    # ITEM 2: Now lets run several times (chi_sq_rounds) the code. Each round we get a chi_sq value based on the sampled obs. frequency. Lets use the average of these calculations.
    for i in range(1, chi_sq_rounds, 1):
        df_standard = (df_inicial_maior_que_minimo.groupby('unit', group_keys=False).apply(pd.DataFrame.sample, n=num_lessons, replace=False)).copy()
        df_final = get_observed_frequency(df_standard, positional_digit)
        df_chi_sq = df_chi_sq.add(df_final.iloc[:, df_final.shape[1] - 1])

    # ITEM 3: Now just dived the sum of the calculated chi_sq by the number of rounds we ran the code.
    # IMPORTANT: The chi_sq for the "Aggregated" unit is not a good indicator, since it may be distorted by the great number os lessons used to generate it.
    df_chi_sq_final = df_chi_sq.div(chi_sq_rounds)
    # Rename column
    df_chi_sq_final.rename(f"chi_sq {chi_sq_rounds} rounds", inplace=True)

    ###################################################
    # GENERATE RESULTS: SPREADSHEET
    ###################################################

    if(positional_digit == 1):
        df_final_concat = pd.concat([df_final, df_chi_sq_final], axis=1)
        # Remove unnecessary row with duplicated aggregated.
        if(type_analysis == 1):
            df_final_concat.drop('to_be_deleted', inplace=True)
        else:
            pass
        # Export results to excel
        df_final_concat.to_excel(writer, sheet_name='D1')
        # Store de final results in a global variable that can be called later.
        global results_D1
        results_D1 = df_final_concat.copy()
    elif(positional_digit == 2):
        df_final_concat = pd.concat([df_final, df_chi_sq_final], axis=1)
        # Remove unnecessary row with duplicated aggregated.
        if (type_analysis == 1):
            df_final_concat.drop('to_be_deleted', inplace=True)
        else:
            pass
        # Export results to excel
        df_final_concat.to_excel(writer, sheet_name='D2')
        # Store de final results in a global variable that can be called later.
        global results_D2
        results_D2 = df_final_concat.copy()
    elif(positional_digit == 3):
        df_final_concat = pd.concat([df_final, df_chi_sq_final], axis=1)
        # Remove unnecessary row with duplicated aggregated.
        if(type_analysis == 1):
            df_final_concat.drop('to_be_deleted', inplace=True)
        else:
            pass
        # Export results to excel
        df_final_concat.to_excel(writer, sheet_name='D3')
        # Store de final results in a global variable that can be called later.
        global results_D3
        results_D3 = df_final_concat.copy()
    else:
        print("\n WARNING! Invalid Digit \n")

    ###################################################
    # GENERATE RESULTS: GRAPHS (Call Function)
    ###################################################

    # Step 1: Generate a Series with ordered values of the chi_sq.
    serie_chi_sq = df_chi_sq_final.sort_values().copy()

    # Step 2: Lets create a list with the name of every unit with the Best and the Worst chi_sq results.
    melhores_chi_sq = serie_chi_sq.head(num_graphs[0])
    piores_chi_sq = serie_chi_sq.tail(num_graphs[1])
    index_melhores_chi_sq = list(melhores_chi_sq.index.values)
    index_piores_chi_sq = list(piores_chi_sq.index.values)
    lista_units_grafico = index_melhores_chi_sq + index_piores_chi_sq + ['_Aggregated']

    # Step 3: Get only unique values from this list.
    lista_units_grafico_unica = list(np.unique(lista_units_grafico))
    # Remove from list the theorical Benford, since we don't want to plot it.
    lista_units_grafico_unica.remove(f"_Benford_D{positional_digit}")

    # Step 4: Now, for every unit (plus the aggregated one) lets create a graph comparing observed and theoretical frequencies.
    for unit_plotada in lista_units_grafico_unica:
        if (positional_digit == 1):
            titulo_chi_sq = round(df_chi_sq_final.loc[unit_plotada], 2)
            titulo_lessons = int(df_final.loc[unit_plotada, "lessons"])
            create_graph(df_final, positional_digit, unit_plotada, titulo_chi_sq, titulo_lessons, chi_sq_rounds, graph_path)
        elif (positional_digit == 2):
            titulo_chi_sq = round(df_chi_sq_final.loc[unit_plotada], 2)
            titulo_lessons = int(df_final.loc[unit_plotada, "lessons"])
            create_graph(df_final, positional_digit, unit_plotada, titulo_chi_sq, titulo_lessons, chi_sq_rounds, graph_path)
        elif (positional_digit == 3):
            titulo_chi_sq = round(df_chi_sq_final.loc[unit_plotada], 2)
            titulo_lessons = int(df_final.loc[unit_plotada, "lessons"])
            create_graph(df_final, positional_digit, unit_plotada, titulo_chi_sq, titulo_lessons, chi_sq_rounds, graph_path)
        else:
            print("\n WARNING! Invalid Digit \n")

    return None

############################################################################################################################################################################################################

def benford(df_inicial, type_analysis, chi_sq_rounds, lessons_D1, lessons_D2, lessons_D3, graphs_worst, graphs_best, xlsx_path, graph_path):
    '''
    :param df: DataFrame with the observed values. Must have one column named "unit" (object) and other column named "value" (numeric).
    :param type_analysis: 1 if the analysis ONLY focus on the entire database as a whole. Anything else if the analysis run over each grouped unit in the database.
    :param chi_sq_rounds: How many times this code ran in order to generate a chi-squared value. This value is the simple mean from the several runs.
    :param lessons_D1, lessons_D2, lessons_D3: Minimum number of lessons per unit considering D1, D2, D3. If some unit has more than the min so we select a sample with the min. This depends on the positional digit.
    :param graphs_worst, graphs_best: How many graphs to be created. num_graphs[0] = number of graphs with the highest chi_sq. num_graphs[1] = number of graphs with worst chi_sq.
    :param xlsx_path: path to save the excel file with the results.
    :param graph_path: Path for the folder where we want to save the graphs.
    return: Main Function. It will call other functions to generate graphs and spreadsheets with the results for D1, D2, D3 analysis.
    '''

    # Make a copy of the initial dataFrame
    df = df_inicial.copy()

    ###################################################
    # CHECKS
    ###################################################

    # 1) DATA FRAME NUM. OF COLUMNS
    if(df.shape[1] != 2):
        print("WARNING! The dataset must have only 2 columns \n")
        sys.exit()
    else:
        print("CHECK 1: PASS! The data has 2 columns!\n")

    # 2) DATA FRAME STRUCTURE OF COLUMNS
    if(df.dtypes[0] == "object" and df.dtypes[1] != "object"):
        df.columns = ['unit', 'value']
        df['value'].astype(float)
        print("CHECK 2: PASS! The one numeric and other text column.\n")
    elif(df.dtypes[0] != "object" and df.dtypes[1] == "object"):
        df.columns = ['value', 'unit']
        df['value'].astype(float)
        print("CHECK 2: PASS! The data has one numeric and one text column.\n")
    else:
        print("WARNING! Are you using Decimal??The dataset must have one column with type TEXT named 'unit' and the other should be named 'value' and be of type NUMBER.\n")

    # 3) TYPE OF ANALYSIS
    if(type_analysis == 1):
        df["unit"] = 'to_be_deleted'
        print("This code will produce results considering all units from the dataset as a whole (If that applies).\n")
    else:
        print("This code will produce results grouped for every unit in the dataset. Plus an aggregated analysis.\n")

    # 4) CHI-SQUARED
    print(f"This code will produce the chi-squared value based on the average value of {chi_sq_rounds} rounds.\n")

    # 5) NUMBER OF LESSONS
    lista_num_lessons = [1,1,1]
    lista_num_lessons[0] = lessons_D1
    lista_num_lessons[1] = lessons_D2
    lista_num_lessons[2] = lessons_D3
    print(f"This code will select a sample size of {lessons_D1} from units when analysing D1. Then a sample size of {lessons_D2} for D2 and {lessons_D3} for D3.\n")

    # 6) NUMBER OF GRAPHS (For aggregated analysis just one graph is needed)
    num_graphs = [1, 1]
    if (type_analysis == "1"):
        pass
    else:
        num_graphs[0] = graphs_best
        num_graphs[1] = graphs_worst
    print(f"This code will generate {num_graphs[1]} graphs from units with worst chi-squared and {num_graphs[0]} for units with the best chi-squared results.\n")

    print("-------------------------- Now wait. The program starts now!  -----------------------------\n\n")

    ###################################################
    # MAIN
    ###################################################
    start = time.time()

    # Path where the results will be written. Outside the loop in order to write several sheets (For D1, D2, D3).
    writer = pd.ExcelWriter(xlsx_path, engine='xlsxwriter')

    # Run the main function for D1, D2 and D3. Will produce the spreadsheet with the obs. freq. plus generate the graphs.
    for i in range(1,4,1):
        create_results(df, lista_num_lessons[i-1], i, chi_sq_rounds, writer, num_graphs, type_analysis, graph_path)
        print(f"\n-> Running the code for the digit {i} and considering the sample size for each unit equals to: {lista_num_lessons[i-1]}\n")

    # Save the spreadsheet file.
    writer.save()

    # Calculate the time took the code to run
    end = time.time()

    print(f"\n This code took {(end - start)} seconds.")

    return [results_D1, results_D2, results_D3]