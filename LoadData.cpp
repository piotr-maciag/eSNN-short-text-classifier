//
// Created by Piotr Maciąg on 14/06/2021.
//

#include "LoadData.h"

int CountInstances(string fileName) {
    fstream handler;

    handler.open(fileName);

    string line;

    int numInstances = 0;

    while (handler.eof() != true) {

        getline(handler, line);

        if (line != "") {
            numInstances++;
        }

    }

    handler.close();

    return numInstances;
}

void LoadDataset(string filename, Dataset *trainingDataset) {
    fstream handler;


    int datasetSize = CountInstances(filename); // zlicz l. instancji w pliku
    //cout << datasetSize << flush;

    handler.open(filename);

    for (int i = 0; i < datasetSize; i++) {
        string line;
        getline(handler, line);
        stringstream linestream(line);
        string dataPortion;


        if (line != "") {

            int i = 1;
            vector<double> inputValues;
            while (getline(linestream, dataPortion, ';') || getline(linestream, dataPortion, ',')) {
                double value = stod(dataPortion);
                int size = trainingDataset->att.size();
                //cout << size << " " << flush;
                if (size < i) {
                    Attribute at;
                    trainingDataset->att.push_back(at);
                }
                inputValues.push_back(value);
                //cout << inputValues.size() << endl;
                i++;
            }
            trainingDataset->inputValues.push_back(inputValues);
            // cout << endl;
        }
    }

    handler.close();
}

void LoadDatasetClasses(string filename, Dataset *trainingDataset) {
    fstream handler;

    int datasetSize = CountInstances(filename); // zlicz l. instancji w pliku

    handler.open(filename);

    for (int i = 0; i < datasetSize; i++) {
        string line;
        getline(handler, line);
        stringstream linestream(line);
        string dataPortion;

        if (line != "") {

            int i = 0;
            vector<int> classes;
            while (getline(linestream, dataPortion, ',') || getline(linestream, dataPortion, ';')) {
                int value = stod(dataPortion) - 1;
                classes.push_back(value);
                if (trainingDataset->maxClass < value + 1) trainingDataset->maxClass = value + 1;
                if (std::find(trainingDataset->Terms.begin(), trainingDataset->Terms.end(), value) ==
                    trainingDataset->Terms.end()) {
                    trainingDataset->Terms.push_back(value); //
                }
                i++;
            }
            trainingDataset->realClasses.push_back(classes);
        }
    }
    handler.close();
}

double AvgCalculatePrecision(Dataset *testDataset, Dataset *trainingDataset) {
    double avgPrec = 0;
    for (int i = 0; i < testDataset->Terms.size(); i++) {
        int TP = 0;
        int FP = 0;
        int TN = 0;
        int FN = 0;
        for (int j = 0; j < testDataset->predictedClasses.size(); j++) {
            int ct = 0;
            for (int k = 0; k < testDataset->predictedClasses[j].size(); k++) {
                if (testDataset->predictedClasses[j][k] == testDataset->Terms[i]) ct = 1;
            }
            int ct1 = 0;
            for (int k = 0; k < testDataset->realClasses[j].size(); k++) {
                if (testDataset->realClasses[j][k] == testDataset->Terms[i]) ct1 = 1;
            }

            if (ct == 0 && ct1 == 0) { TN++; }
            else if (ct == 1 && ct1 == 1) { TP++; }
            else if (ct == 0 && ct1 == 1) { FN++; }
            else if (ct == 1 && ct1 == 0) { FP++; }
        }

        if (TP == 0 && FP == 0) {
            avgPrec += 0;
        } else {
            avgPrec += double(TP) / (double(TP) + double(FP));
        }
    }
    return avgPrec / testDataset->Terms.size();
}

double AvgCalculateRecall(Dataset *testDataset, Dataset *trainingDataset) {
    double avgRec = 0;
    for (int i = 0; i < testDataset->Terms.size(); i++) {
        int TP = 0;
        int FP = 0;
        int TN = 0;
        int FN = 0;
        for (int j = 0; j < testDataset->predictedClasses.size(); j++) {
            int ct = 0;
            for (int k = 0; k < testDataset->predictedClasses[j].size(); k++) {
                if (testDataset->predictedClasses[j][k] == testDataset->Terms[i]) ct = 1;
            }
            int ct1 = 0;
            for (int k = 0; k < testDataset->realClasses[j].size(); k++) {
                if (testDataset->realClasses[j][k] == testDataset->Terms[i]) ct1 = 1;
            }

            if (ct == 0 && ct1 == 0) { TN++; }
            else if (ct == 1 && ct1 == 1) { TP++; }
            else if (ct == 0 && ct1 == 1) { FN++; }
            else if (ct == 1 && ct1 == 0) { FP++; }
        }

        if (TP == 0 && FN == 0) {
            avgRec += 0;
        } else {
            avgRec += double(TP) / (double(TP) + double(FN));
        }
    }
    return avgRec / testDataset->Terms.size();
}

double AvgFMeasure(Dataset *testDataset, Dataset *trainingDataset) {
    double avgRec;
    double avgPrec;
    for (int i = 1; i <= trainingDataset->maxClass; i++) {
        int TP = 0;
        int FP = 0;
        int TN = 0;
        int FN = 0;
        for (int j = 0; j < testDataset->predictedClasses.size(); j++) {
            int ct = 0;
            for (int k = 0; k < testDataset->predictedClasses[j].size(); k++) {
                if (testDataset->predictedClasses[j][k] == i) ct = 1;
            }
            int ct1 = 0;
            for (int k = 0; k < testDataset->realClasses[j].size(); k++) {
                if (testDataset->realClasses[j][k] == i) ct1 = 1;
            }

            if (ct == 0 && ct1 == 0) { TN++; }
            else if (ct == 1 && ct1 == 1) { TP++; }
            else if (ct == 0 && ct1 == 1) { FN++; }
            else if (ct == 1 && ct1 == 0) { FP++; }
        }

        avgRec += double(TP) / (double(TP) + double(FN));
        avgPrec += (TP) / (TP + FP);
    }
    return avgRec / trainingDataset->maxClass;
}

double Acc(Dataset *testDataset) {
    double P = 0;
    for (int i = 0; i < testDataset->predictedClasses.size(); i++) {
        for (int j = 0; j < testDataset->realClasses[i].size(); j++) {
            if (testDataset->realClasses[i][j] == testDataset->predictedClasses[i][j]) P += 1;
            //cout << testDataset->realClasses[i][j] << " " << testDataset->predictedClasses[i][j] << endl;
        }
    }
    return (P / double(testDataset->inputValues.size()));
}

void PrintDataset(Dataset *d) {

    for (int i = 0; i < d->inputValues.size(); i++) {
        for (int j = 0; j < d->inputValues[i].size(); j++) {
            cout << d->inputValues[i][j] << ',';
        }

        for (int j = 0; j < d->realClasses[i].size(); j++) {
            cout << d->realClasses[i][j] << ',';
        }
        cout << endl;
    }

}

//Clear all structures after each eSNN training and classification
//void ClearStructures() {
//
//    for (int i = 0; i < OutputNeurons.size(); i++) {
//        for(int j = 0; j < OutputNeurons[i].size(); j++)
//            delete OutputNeurons[i][j];
//    }
//
//    OutputNeurons.clear();
//    X.clear();
//    Y.clear();
//
//    for (int k = 0; k < InputNeurons.size(); k++) {
//        for (int j = 0; j < InputNeurons.size(); j++) {
//            delete InputNeurons[k][j];
//        }
//    }
//
//    InputNeurons.clear();
//    Wstream.clear();
//    I_min.clear();
//    I_max.clear();
//    IDS.clear();
//}

//void SaveResults(string path, Dataset * d)
//{
//    fstream results;
//    results.open(path, fstream::out);
//
//    for(int i = 0; i < d->realValues.size(); i++)
//    {
//        results << d->realValues[i] << "," << d->predictedValues[i] << endl;
//    }
//
//    results.close();
//}

void PrintResults(Dataset * d, string fileName)
{
    fstream results;
    results.open(fileName, fstream::out);
    for(int i = 0 ; i< d->predictedClasses.size(); i++)
    {
        for(int j = 0; j < d->predictedClasses[i].size(); j++)
        {
            results << d->predictedClasses[i][j] << "," << d->realClasses[i][j];
        }
        results << endl;
    }

    results.close();
}