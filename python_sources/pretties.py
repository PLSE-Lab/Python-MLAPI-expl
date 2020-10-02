import IPython.display as ip_disp
import pandas as pd
import numpy as np


def max_data_frame_columns(n=None):
    '''
    Configura o ambiente pandas para mostrar n colunas. Passando o valor None todas as colunas serão mostradas.
    Default n: None
    :param n:
    :return:
    '''
    pd.set_option('display.max_columns', n)

def decimal_notation(round_to=2):
    pd.set_option('display.float_format', lambda x: '%.{}f'.format(round_to) % x)

def display(element):
    #https://stackoverflow.com/questions/26873127/show-dataframe-as-table-in-ipython-notebook
    try:
        ip_disp.display(ip_disp.HTML(element.to_html()))
    except AttributeError:
        ip_disp.display(element)

# def columm_name_descrition(columm_name, description):
#     ip_disp.display(ip_disp.Markdown("# <font color='red'>{}</font>".format(columm_name)))
#     ip_disp.display(ip_disp.Markdown(description))

def display_html(html):
    ip_disp.display(ip_disp.HTML(html))
 
def display_md(md):
    ip_disp.display(ip_disp.Markdown(md))

def group_colors(column, color_map):
    return column.index.map(lambda value: 'background-color: {}'.format(color_map[value]))

def group_colors_map(df, group_colors_by, color_1='#ccedd2', color_2='#effcef', sort_rows_by_grouped_column=True):
    '''
    Mapeia cada agrupamento para uma cor.
    As cores são distintas para grupos adjacentes (intercalando), então a ordem do pandas.DataFrame importa.

    Exemplo 1)
        Considere a sequencia [A, A, A, B, B, C, C, D, E, E]
        Serão associadas à color_1 os elementos A, C e E
        Serão associadas à color_2 os elementos B e D
        A sequencia de cores será  [color_1, color_1, color_1,
                                    color_2, color_2,
                                    color_1, color_1,
                                    color_2,
                                    color_1, color_1]

    Serão pintados os grupos de elementos que estão adjacentes e no mesmo grupo. De forma que mesmo elementos
    iguais, que não estão adjacentes, podem eventualmente ter cores distintas:
    Exemplo2)

        Considere a sequencia [A, A, B, B, C, C, A, D, E, E]
        Serão associadas à color_1 os elementos o dois primeiros A, os C e o D
        Serão associadas à color_2 os elementos B, o terceiro A e o E
        A sequencia de cores será  [color_1, color_1,
                                    color_2, color_2,
                                    color_1, color_1,
                                    color_2,
                                    color_1,
                                    color_2, color_2]

    Dica: ordenando os elementos dos Exemplo 2) chegamos no Exemplo 1)
    Marque o parametro sort_rows_by_grouped_column para True.


    :param df:
    :param group_colors_by: coluna de referência para fazer os agrupamentos
    :param color_1:
    :param color_2:
    :return:
    '''


    if sort_rows_by_grouped_column:
        df = df.sort_values(by=group_colors_by)

    groups_len = df.groupby(group_colors_by).apply(len)

    index_colors = {}
    use_color_1 = True

    for group_label in groups_len.index:
        use_color = color_1 if use_color_1 else color_2

        for i in df[df[group_colors_by] == group_label].index:
            index_colors[i] = use_color

        use_color_1 = not use_color_1

    return index_colors

def paint_df_grouped_by(df, column):
    '''
    Mostra o pandas.DataFrame com as linhas pintadas mantendo a mesma cor para linhas adjacentes que possuam mesmo
    valor para a variável column.
    :param df:
    :param column:
    :return:
    '''
    color_map = group_colors_map(df, column)
    df_painted = df.style.apply(lambda column : group_colors(column, color_map))
    ip_disp.display(df_painted)


def walk_dict(a_dict, path=["."], apply_func=None, print_leaf=False):
    for k in a_dict.keys():
        if type(a_dict[k]) is dict:
            print(str([[element] for element in path]).replace(", ", "")[1:-1])
            if apply_func:
                print(k.upper(), apply_func(a_dict[k]))
            else:
                print(k.upper(), "memory:", sys.getsizeof(a_dict[k]))
            print()
            path.append(k)
            walk_dict(a_dict[k], path)
        elif print_leaf:
            print(str([[element] for element in path] + [k]).replace(", ", "")[1:-1])
            print("\t", "LEAF", k.upper(), "memory:", sys.getsizeof(a_dict[k]))
            print()
    path.pop()
