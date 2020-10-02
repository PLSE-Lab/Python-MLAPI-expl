def file_to_list1(file_name):

    my_lst = []

    with open(file_name, 'r', encoding="utf8") as input_file:
        for line in input_file:
            columns = line.rstrip().split(",")
            columns = [i.replace('"', '') for i in columns]
            columns = [i.replace(' ', '') for i in columns]
            my_lst.append(columns)

    return my_lst

def file_to_list2(file_name):

    my_lst = []

    with open(file_name, 'r', encoding="utf8") as input_file:
        for line in input_file:
            columns = line.rstrip().split(",")
            # columns = [i.replace('"', '') for i in columns]
            # columns = [i.replace(' ', '') for i in columns]
            my_lst.append(columns)

    return my_lst

def write_to_file(filename, result):
    with open(filename,'w') as f:
        for line in result:
            f.write(str(line[0]-1) + ',' + '"' + str(line[1]) + '"' + '\n')

def main(my_list, spam_list):
    my_dict = {}
    final_out = []
    for i in range(1,len(my_list)):
        for n in range(1,len(my_list[i])):
            if my_list[i][n] not in my_dict:
                my_dict[my_list[i][n]] = my_list[i][0]
    
    for i in range(1,len(spam_list)):
        out = []
        temp = spam_list[i][1].split()
        for n in range(len(temp)):
            if temp[n].lower() in my_dict and my_dict[temp[n].lower()] not in out:
                out.append(int(my_dict[temp[n].lower()]))
        final_out.append([i, out])
        
    return final_out

my_list = file_to_list1("Extra Material 2 - keyword list_with substring.csv")
spam_list = file_to_list2("Keyword_spam_question.csv")
result = main(my_list, spam_list)
write_to_file("test.csv", result)
print(main)