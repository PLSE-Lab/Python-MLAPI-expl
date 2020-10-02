import os
import time
import pandas as pd
import matplotlib.pyplot as plt


CITY_DATA = { 'chicago': 'chicago.csv',
              'new york city': 'new_york_city.csv',
              'washington': 'washington.csv' }
city_list=['chicago','new york city','washington']
month_list=['january','february','march','april','may','june']
month_title_list=['January','February','March','April','May','June']
weekday_name_list=['monday','tuesday','wednesday','thursday','friday','saturday','sunday']
weekday_title_list=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

def get_filters():
    # get user input for city (chicago, new york city, washington).
    while True:
        city=input("Which city would like to explore?  Please enter 'chicago', 'new york city' or 'washington'\n")
        city=city.lower()
        if city in city_list:
            break
        print("\nInvalid input.\n")
    # get user input for month (all, january, february, ... , june)
    while True:
        month=input('Which month do you want to look at? january,february,march,april,may,june, or all? \n')
        month=month.lower()
        if month=='all' or month in month_list:
            break
        print("\nInvalid input.\n")
    # get user input for day of week (all, monday, tuesday, ... sunday)
    while True:
        day=input('Which day of week do you want to look at? sunday, monday, tuesday, wednesday, thursday, friday, ,saturday or all?\n')
        day=day.lower()
        if day=='all' or day in weekday_name_list:
            break
        print("\nInvalid input.\n")
    print('-'*40)
    return city, month, day

def load_data(city, month, day):
    '''get the data the user asked for'''
    df=pd.read_csv(CITY_DATA[city])
    df['Start Time']=pd.to_datetime(df['Start Time'])
    df['month']=df['Start Time'].dt.month
    df['day of week']=df['Start Time'].dt.dayofweek
    if month!='all':
        month = month_list.index(month)+1
        df=df[df['month']==month]
    if day!='all':
        day=weekday_name_list.index(day)
        df = df[df['day of week'] == day]
    return df

def trip_duration_stats(city, df):
    """Displays statistics on the total and average trip duration."""
    print('\nCalculating Trip Duration of the city {}...\n'.format(city))
    start_time = time.time()
    total_duration=df['Trip Duration'].sum()
    average_duration=df['Trip Duration'].mean()
    result='The total and average trip durations are {} seconds and {} seconds\n'.format(total_duration,average_duration)
    print(result)
    print("\nThis took %s seconds." % (time.time() - start_time))
    print('-'*40)
    return result

def station_stats(city,df):
    """Displays statistics on the most popular stations and trip."""
    print('\nCalculating The Most Popular Stations and Trip of the city {}...\n'.format(city))
    start_time = time.time()
    # display most commonly used start station
    str1='The most common start station is {}.'.format(df['Start Station'].mode()[0])
    # TO DO: display most commonly used end station
    str2='\nThe most common end station is {}.'.format(df['End Station'].mode()[0])

    # display most frequent combination of start station and end station trip
    df['trip'] = df['Start Station'] + ' to ' + df['End Station']
    str3='\nThe most common combination of start and end station trip is {}.\n'.format(df['trip'].mode()[0])
    print(str1+str2+str3)
    print("\nThis took %s seconds." % (time.time() - start_time))
    print('-'*40)
    return str1+str2+str3

def time_stats(city,df):
    """Displays statistics on the most frequent times of travel."""
    print('\nCalculating The Most Frequent Times of Travel of the city {}...\n'.format(city))
    start_time = time.time()
    # display the most common month
    df['Start Time']=pd.to_datetime(df['Start Time'])
    df['month']=df['Start Time'].dt.month
    common_month=month_list[df['month'].mode()[0]-1]
    str1='The most common month is {}.'.format(common_month.title())
    df['day of week']=df['Start Time'].dt.dayofweek
    common_dayofweek=weekday_title_list[df['day of week'].mode()[0]]
    str2='\nThe most common day of week is {}.'.format(common_dayofweek)
    df['Start Hour']=df['Start Time'].dt.hour
    str3='\nThe most common start hour is {}:00\n'.format(df['Start Hour'].mode()[0])
    print(str1+str2+str3)
    print("\nThis took %s seconds." % (time.time() - start_time))
    print('-'*40)
    return str1+str2+str3

def weekday_trip_counts(city, df):
    '''Calculates the total number of trips of each weekday'''
    print('\nCalculating total trip counts of each weekday of the city {}...\n'.format(city))
    start_time = time.time()
    dayofweek_df=df.groupby(['day of week'])
    result=dayofweek_df['day of week'].count()
    if result.size==7:
        result.index=weekday_title_list
    else:
        result.index=[weekday_title_list[result.index[0]]]
    print(result)
    print("\nThis took %s seconds." % (time.time() - start_time))
    print('-'*40)
    return result

def monthly_trip_counts(city,df):
    '''Calculates the number of trips of each month'''
    print('\nCalculating monthly trip counts of the city {}...\n'.format(city))
    start_time = time.time()
    monthly_df=df.groupby('month')
    result=monthly_df['Start Time'].count()
    if result.size==6:
        result.index=month_title_list
    else:
        result.index=[month_title_list[result.index[0]-1]]
    print(result)
    print("\nThis took %s seconds." % (time.time() - start_time))
    print('-'*40)
    return result

def monthly_mean_duration(city,df):
    '''Calculates the mean trip duration of each month'''
    print('\nCalculating monthly mean trip duration of the city {}...\n'.format(city))
    start_time = time.time()
    grouped_monthly=df.groupby('month')
    result=grouped_monthly['Trip Duration'].mean()
    if result.size==6:
        result.index=month_title_list
    else:
        result.index=[month_title_list[result.index[0]-1]]
    print(result)
    print("\nThis took %s seconds." % (time.time() - start_time))
    print('-'*40)
    return result

def weekday_mean_duration(city,df):
    '''Calculates the mean trip duration of each weekday'''
    print('\nCalculating weekday mean trip duration of the city {}...\n'.format(city))
    start_time = time.time()
    groupedby_dayofweek=df.groupby('day of week')
    result=groupedby_dayofweek['Trip Duration'].mean()
    if result.size==7:
        result.index=weekday_title_list
    else:
        result.index=[weekday_title_list[result.index[0]]]
    print(result)
    print("\nThis took %s seconds." % (time.time() - start_time))
    print('-'*40)
    return result

def birth_year_stats(city,df):
    '''Display the most common, oldest and yougest user birth years.'''
    if 'Birth Year' in df.columns:
        print('\nCalculating user birth year stats of the city {}...\n'.format(city))
        start_time = time.time()
        str1='The most common birth year is {}.'.format(df['Birth Year'].mode()[0])
        str2='\nThe oldest birth year is {}.'.format(df['Birth Year'].min())
        str3='\nThe youngest birth year is {}.\n'.format(df['Birth Year'].max())
        print(str1+str2+str3)
        print("\nThis took %s seconds." % (time.time() - start_time))
        print('-'*40)
        return str1+str2+str3
    else:
        return []

def birth_year_distribution(city,df):
    '''Calculates user birth year distribution'''
    if 'Birth Year' in df.columns:
        print('\nCalculating the distribution of user birth year of the city {}...\n'.format(city))
        start_time = time.time()
        grouped_birthyear=df.groupby('Birth Year')
        result=grouped_birthyear['Trip Duration'].count()
        print(result.head())
        print("\nThis took %s seconds." % (time.time() - start_time))
        print('-'*40)
        return result
    return []

def user_type_stats(city,df):
    '''Calculates user type stats'''
    print('\nCalculating user type stats of the city {}...\n'.format(city))
    start_time = time.time()
    result=df['User Type'].value_counts()
    print(result)
    print("\nThis took %s seconds." % (time.time() - start_time))
    print('-'*40)
    return result

def user_gender_stats(city,df):
    '''Calculates user gender stats'''
    if 'Gender' in df.columns:
        print('\nCalculating user gender stats of the city {}...\n'.format(city))
        start_time = time.time()
        result=df['Gender'].value_counts()
        print(result)
        print("\nThis took %s seconds." % (time.time() - start_time))
        print('-'*40)
        return result
    return []

def write_to_file(output, path, title, txt_file):
    '''write the results of stats inquiry to files'''
    plt.ion()
    if type(output) is str:
        print(output,file=txt_file)
    elif len(output)!=0:
        output.to_string(txt_file)
        print('\n', file=txt_file)
        if len(output)<10:
            output.plot.bar()
            plt.xticks(rotation=45)
        else:
            output.plot()
        plt.title(title)
        plt.savefig('{}\\{}.png'.format(path,title), bbox_inches='tight')
        plt.show()
        plt.pause(1)
        plt.close()
    else:
        print('This city does not have the corresponding data\n', file=txt_file)

def calculate_all(path,txt_file):
    '''calculte the stats of all three cities'''
    city_dataframe={}
    for city in CITY_DATA:
        city_dataframe[city]=pd.read_csv(CITY_DATA[city])
        city_dataframe[city]['Start Time']=pd.to_datetime(city_dataframe[city]['Start Time'])
        city_dataframe[city]['month']=city_dataframe[city]['Start Time'].dt.month
        city_dataframe[city]['date']=city_dataframe[city]['Start Time'].dt.date
        city_dataframe[city]['day of week']=city_dataframe[city]['Start Time'].dt.dayofweek
    for f in function_list:
        for city in city_dataframe:
            title='{} of the city {}'.format(f.__name__,city)
            print(title, file=txt_file)
            output=f(city,city_dataframe[city])
            write_to_file(output, path, title, txt_file)


def single_city(city,df,path,txt_file):
    '''calculates the stats of a single city'''
    for f in function_list:
        title='{} of the city {}'.format(f.__name__,city)
        print(title, file=txt_file)
        output=f(city,df)
        write_to_file(output, path, title, txt_file)

function_list=[trip_duration_stats,station_stats,time_stats,weekday_trip_counts,
                monthly_trip_counts,monthly_mean_duration,weekday_mean_duration,
                birth_year_stats,birth_year_distribution,user_type_stats,user_gender_stats]

def createfilepath():
    '''create a folder to save inquiry results'''
    current_time=str(time.ctime())
    folder_name=current_time.replace(" ", "_")
    folder_name=folder_name.replace(":", "_")
    os.mkdir(folder_name)
    new_path=os.path.join(os.getcwd(),folder_name)
    txt_file = open("{}\\{}.txt".format(new_path, folder_name), 'w')
    files_saved_to="\nResults of this inquiry is save in the folder {}\n".format(folder_name)
    print(files_saved_to)
    return new_path, txt_file, files_saved_to

def display_raw_data(df):
    '''display raw data on screen'''
    n=5
    print(df.head(n))
    while True:
        more=input("\nWould you like to see more raw data?  Please eneter 'yes' or 'no'\n").lower()
        if more =='no':
            break
        elif more=='yes':
            n=n+5
            print(df.iloc[n-5:n, :])
        else:
            print('\nInvalid input')

def main():
    print("\nHello! Let\'s explore some US bikeshare data!")
    while True:
        while True:
            raw_or_stats=input("\nWould you like to see raw data or stats?   Please enter 'raw' or 'stats'\n").lower()
            if raw_or_stats=='raw':
                city, month, day = get_filters()
                df = load_data(city, month, day)
                display_raw_data(df)
                break
            elif raw_or_stats =='stats':
                new_path, txt_file, files_saved_to = createfilepath()
                while True:
                    all_cities=input("Would you like to stats of all three cities?  Please enter 'yes' or 'no'\n").lower()
                    if all_cities=='yes':#the user like to view all cities
                            #calculates various stats of all three cities
                        calculate_all(new_path,txt_file)
                        break
                    elif all_cities=='no':
                        city, month, day = get_filters()
                        df = load_data(city, month, day)
                        single_city(city,df,new_path,txt_file)
                        break
                    else:
                        print('Invalid input.\n')
                print(files_saved_to)
                txt_file.close()
                break
            else:
                print('\nInvalid input')
        restart = input("Would you like to restart? Please enter 'yes' or 'no'.\n").lower()
        if restart.lower() != 'yes':
            break

if __name__ == "__main__":
	main()
