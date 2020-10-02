#include <math.h>
#include <iomanip>
#include <string.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <map>
#include <fstream>
#include <regex>
using namespace std;

float Output(float actual)
{

    float output = (float) (1.0f/(1.0f + exp((float)-actual)));
    output = max( (float)1e-15f, min( (float)1-1e-15f, output ) );
    return output;
}

string string_replace( string src, string const& target, string const& repl)
{
    // handle error situations/trivial cases

    if (target.length() == 0) {
        // searching for a match to the empty string will result in
        //  an infinite loop
        //  it might make sense to throw an exception for this case
        return src;
    }

    if (src.length() == 0) {
        return src;  // nothing to match against
    }

    size_t idx = 0;

    for (;;) {
        idx = src.find( target, idx);
        if (idx == string::npos)  break;

        src.replace( idx, target.length(), repl);
        idx += repl.length();
    }

    return src;
}

vector<string> tokenizeString(const char* src,
                             char delim,
                             bool want_empty_tokens)
{
    vector<string> tokens;


    if (src and *src != '\0') // defensive
        while( true )  {
            if(*src == '\"')
            {

                const char* d = strchr(src+1, '\"')+1;
                size_t len = (d)?d-src: strlen(src);

                if (len or want_empty_tokens)
                {
                    string s = string(src, len);
                    s.erase(s.find_last_not_of(",\n\r\t")+1);
                    tokens.push_back(s);// capture token
                }

                if (d!=NULL &&strlen(d)!=0)
                    src += len+1;
                else
                    break;
            }
            else
            {
                const char* d = strchr(src+1, delim);
                size_t len = (d)? d-src : strlen(src);

                if (len or want_empty_tokens)
                {
                    string s = string(src, len);
                    s.erase(s.find_last_not_of("\n\r\t")+1);
                    tokens.push_back(s);// capture token

                }

                if (d!=NULL && strlen(d)!=0)
                    src += len+1;
                else
                    break;
            }



        }

    return tokens;
}

//Fitness Functions
float fitPrediction(map<string,vector<float>>& data , unsigned int row)
{
    float p = (min((float) ((((0.058823499828577f + data["Sex"][row]) - cos((data["Pclass"][row] / 2.0f))) * 2.0f)), (float) ((0.885868))) * 2.0f) +
              max((float) ((data["SibSp"][row] - 2.409090042114258f)), (float) ( -(min((float) (data["Sex"][row]), (float) (sin(data["Parch"][row]))) * data["Pclass"][row]))) +
              (0.138462007045746f * ((min((float) (data["Sex"][row]), (float) (((data["Parch"][row] / 2.0f) / 2.0f))) * data["Age"][row]) - data["Cabin"][row])) +
              min((float) ((sin((data["Parch"][row] * ((data["Fare"][row] - 0.720430016517639f) * 2.0f))) * 2.0f)), (float) ((data["SibSp"][row] / 2.0f))) +
              max((float) (min((float) ( -cos(data["Embarked"][row])), (float) (0.138462007045746f))), (float) (sin(((data["Cabin"][row] - data["Fare"][row]) * 2.0f)))) +
              -min((float) ((((data["Age"][row] * data["Parch"][row]) * data["Embarked"][row]) + data["Parch"][row])), (float) (sin(data["Pclass"][row]))) +
              min((float) (data["Sex"][row]), (float) ((sin( -(data["Fare"][row] * cos((data["Fare"][row] * 1.630429983139038f)))) / 2.0f))) +
              min((float) ((0.230145)), (float) (sin(min((float) (((67.0f / 2.0f) * sin(data["Fare"][row]))), (float) (0.31830988618379069f))))) +
              sin((sin(data["Cabin"][row]) * (sin((12.6275)) * max((float) (data["Age"][row]), (float) (data["Fare"][row]))))) +
              sin(((min((float) (data["Fare"][row]), (float) ((data["Cabin"][row] * data["Embarked"][row]))) / 2.0f) *  -data["Fare"][row])) +
              min((float) (((2.675679922103882f * data["SibSp"][row]) * sin(((96) * sin(data["Cabin"][row]))))), (float) (data["Parch"][row])) +
              sin(sin((max((float) (min((float) (data["Age"][row]), (float) (data["Cabin"][row]))), (float) ((data["Fare"][row] * 0.31830988618379069f))) * data["Cabin"][row]))) +
              max((float) (sin(((12.4148) * (data["Age"][row] / 2.0f)))), (float) (sin((-3.0f * data["Cabin"][row])))) +
              (min((float) (sin((((sin(((data["Fare"][row] * 2.0f) * 2.0f)) * 2.0f) * 2.0f) * 2.0f))), (float) (data["SibSp"][row])) / 2.0f) +
              ((data["Sex"][row] - data["SibSp"][row]) * (cos(((data["Embarked"][row] - 0.730768978595734f) + data["Age"][row])) / 2.0f)) +
              ((sin(data["Cabin"][row]) / 2.0f) - (cos(min((float) (data["Age"][row]), (float) (data["Embarked"][row]))) * sin(data["Embarked"][row]))) +
              min((float) (0.31830988618379069f), (float) ((data["Sex"][row] * (2.212120056152344f * (0.720430016517639f - sin((data["Age"][row] * 2.0f))))))) +
              (min((float) (cos(data["Fare"][row])), (float) (max((float) (sin(data["Age"][row])), (float) (data["Parch"][row])))) * cos((data["Fare"][row] / 2.0f))) +
              sin((data["Parch"][row] * min((float) ((data["Age"][row] - 1.5707963267948966f)), (float) ((cos((data["Pclass"][row] * 2.0f)) / 2.0f))))) +
              (data["Parch"][row] * (sin(((data["Fare"][row] * (0.623655974864960f * data["Age"][row])) * 2.0f)) / 2.0f)) +
              (0.31830988618379069f * cos(max((float) ((0.602940976619720f * data["Fare"][row])), (float) ((sin(0.720430016517639f) * data["Age"][row]))))) +
              (min((float) ((data["SibSp"][row] / 2.0f)), (float) (sin(((data["Pclass"][row] - data["Fare"][row]) * data["SibSp"][row])))) * data["SibSp"][row]) +
              tanh((data["Sex"][row] * sin((5.199999809265137f * sin((data["Cabin"][row] * cos(data["Fare"][row]))))))) +
              (min((float) (data["Parch"][row]), (float) (data["Sex"][row])) * cos(max((float) ((cos(data["Parch"][row]) + data["Age"][row])), (float) (3.1415926535897931f)))) +
              (min((float) (tanh(((data["Cabin"][row] / 2.0f) + data["Parch"][row]))), (float) ((data["Sex"][row] + cos(data["Age"][row])))) / 2.0f) +
              (sin((sin(data["Sex"][row]) * (sin((data["Age"][row] * data["Pclass"][row])) * data["Pclass"][row]))) / 2.0f) +
              (data["Sex"][row] * (cos(((data["Sex"][row] + data["Fare"][row]) * ((8.48635) * (63)))) / 2.0f)) +
              min((float) (data["Sex"][row]), (float) ((cos((data["Age"][row] * tanh(sin(cos(data["Fare"][row]))))) / 2.0f))) +
              (tanh(tanh( -cos((max((float) (cos(data["Fare"][row])), (float) (0.094339601695538f)) * data["Age"][row])))) / 2.0f) +
              (tanh(cos((cos(data["Age"][row]) + (data["Age"][row] + min((float) (data["Fare"][row]), (float) (data["Age"][row])))))) / 2.0f) +
              (tanh(cos((data["Age"][row] * ((-2.0f + sin(data["SibSp"][row])) + data["Fare"][row])))) / 2.0f) +
              (min((float) (((281) - data["Fare"][row])), (float) (sin((max((float) ((176)), (float) (data["Fare"][row])) * data["SibSp"][row])))) * 2.0f) +
              sin(((max((float) (data["Embarked"][row]), (float) (data["Age"][row])) * 2.0f) * (((785) * 3.1415926535897931f) * data["Age"][row]))) +
              min((float) (data["Sex"][row]), (float) (sin( -(min((float) ((data["Cabin"][row] / 2.0f)), (float) (data["SibSp"][row])) * (data["Fare"][row] / 2.0f))))) +
              sin(sin((data["Cabin"][row] * (data["Embarked"][row] + (tanh( -data["Age"][row]) + data["Fare"][row]))))) +
              (cos(cos(data["Fare"][row])) * (sin((data["Embarked"][row] - ((734) * data["Fare"][row]))) / 2.0f)) +
              ((min((float) (data["SibSp"][row]), (float) (cos(data["Fare"][row]))) * cos(data["SibSp"][row])) * sin((data["Age"][row] / 2.0f))) +
              (sin((sin((data["SibSp"][row] * cos((data["Fare"][row] * 2.0f)))) + (data["Cabin"][row] * 2.0f))) / 2.0f) +
              (((data["Sex"][row] * data["SibSp"][row]) * sin(sin( -(data["Fare"][row] * data["Cabin"][row])))) * 2.0f) +
              (sin((data["SibSp"][row] * ((((5.428569793701172f + 67.0f) * 2.0f) / 2.0f) * data["Age"][row]))) / 2.0f) +
              (data["Pclass"][row] * (sin(((data["Embarked"][row] * data["Cabin"][row]) * (data["Age"][row] - (1.07241)))) / 2.0f)) +
              (cos((((( -data["SibSp"][row] + data["Age"][row]) + data["Parch"][row]) * data["Embarked"][row]) / 2.0f)) / 2.0f) +
              (0.31830988618379069f * sin(((data["Age"][row] * ((data["Embarked"][row] * sin(data["Fare"][row])) * 2.0f)) * 2.0f))) +
              ((min((float) ((data["Age"][row] * 0.058823499828577f)), (float) (data["Sex"][row])) - 0.63661977236758138f) * tanh(sin(data["Pclass"][row]))) +
              -min((float) ((cos(((727) * ((data["Fare"][row] + data["Parch"][row]) * 2.0f))) / 2.0f)), (float) (data["Fare"][row])) +
              (min((float) (cos(data["Fare"][row])), (float) (data["SibSp"][row])) * min((float) (sin(data["Parch"][row])), (float) (cos((data["Embarked"][row] * 2.0f))))) +
              (min((float) (((data["Fare"][row] / 2.0f) - 2.675679922103882f)), (float) (0.138462007045746f)) * sin((1.5707963267948966f * data["Age"][row]))) +
              min((float) ((0.0821533)), (float) (((sin(data["Fare"][row]) + data["Embarked"][row]) - cos((data["Age"][row] * (9.89287))))));
    if(isnan(p)||!isfinite(p))
    {
        cout << "Error at Row: " << row << " - " << data["signal"][row] << endl;
        return 1;
    }
    return 1-Output(p)>.5?1:0;
}

int main() {

    //Parse training file and label encode specific columns
    string inputfileName = "/home/karl/Development/Kaggle/Titanic/Data/test.csv";
    string outputfileName = "/home/karl/Development/Kaggle/Titanic/gptest.csv";
    string line;
    vector<string> ids;
    map<string, vector<float>> data;
    vector<string> header;
    ifstream stream(inputfileName);
    if(stream.good())
    {
        bool isHeader = true;


        while(getline(stream, line))
        {
            if(isHeader)
            {

                vector<string> rawHeader = tokenizeString(line.c_str(),',',true);
                for(auto& x : rawHeader)
                {
                    header.push_back(x);
                }
                isHeader=false;
            }
            else
            {

                //cout << line << endl;
                line = string_replace(line,",,",", ,");
                line = string_replace(line,"\"\"","");
                //cout << line << endl;
                vector<string> lineData  = tokenizeString(line.c_str(),',',true);

                int column = 0;
                for(auto &x : lineData)
                {

                    //cout << header[column] << endl;
                    //cout << x << endl;
                    if(header[column]=="PassengerId")
                    {
                        ids.push_back(x);
                    }
                    else if(header[column]=="Name"||header[column]=="Ticket")//Ignore
                    {

                    }
                    else if(header[column]=="Sex")
                    {
                        if(x=="male")
                        {
                            data[header[column]].push_back(1);
                        }
                        else
                        {
                            data[header[column]].push_back(0);
                        }


                    }
                    else if(header[column]=="Cabin")
                    {
                        if(x[0]=='A')
                        {
                            data[header[column]].push_back(1);
                        }
                        else if(x[0]=='B')
                        {
                            data[header[column]].push_back(2);
                        }
                        else if(x[0]=='C')
                        {
                            data[header[column]].push_back(3);
                        }
                        else if(x[0]=='D')
                        {
                            data[header[column]].push_back(4);
                        }
                        else if(x[0]=='E')
                        {
                            data[header[column]].push_back(5);
                        }
                        else if(x[0]=='F')
                        {
                            data[header[column]].push_back(6);
                        }
                        else if(x[0]=='G')
                        {
                            data[header[column]].push_back(7);
                        }
                        else if(x[0]=='T')
                        {
                            data[header[column]].push_back(8);
                        }
                        else
                        {
                            data[header[column]].push_back(0);
                        }


                    }
                    else if(header[column]=="Embarked")
                    {
                        if(x[0]=='C')
                        {
                            data[header[column]].push_back(1);
                        }
                        else if(x[0]=='Q')
                        {
                            data[header[column]].push_back(2);
                        }
                        else if(x[0]=='S')
                        {
                            data[header[column]].push_back(3);
                        }
                        else
                        {
                            data[header[column]].push_back(0);
                        }
                    }
                    else
                    {
                        if(x.length()>0&&x[0]!=' ')
                            data[header[column]].push_back(stof(x));
                        else
                            data[header[column]].push_back(-1.0f);
                    }
                    column++;
                }
            }
        }


    }
    else
    {
        cout << "File doesn't exist" << endl;
    }

    ofstream out(outputfileName);
    if(!out) {
        cout << "Cannot open file.\n";
        return 1;
    }

    if(data.count("Survived")==1)
    {
        out << "PassengerId" << "," << "Survived" << "," << "Prediction" << endl;
    }
    else
    {
        out << "PassengerId" << "," << "Survived" << endl;
    }

    unsigned int rows = ids.size();
    cout << rows << endl;
    out << std::setprecision(6);
    out << std::fixed;
    for(unsigned int row = 0; row < rows; row++)
    {
        cout << row << endl;
        float prediction = fitPrediction(data,row);
        if(data.count("Survived")==1)
        {
            out << ids[row] << "," << int(data["Survived"][row]) << "," << prediction << endl;
        }
        else
        {
            out << ids[row] << "," << int(prediction) << endl;
        }
        cout << row << ":" << prediction << endl;

    }

    out.close();
    return 0;
}