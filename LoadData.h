//
// Created by Piotr Maciąg on 14/06/2021.
//

#ifndef PREDICTION_LOADDATA_H
#define PREDICTION_LOADDATA_H

#include "iostream"
#include "algorithm"
#include "vector"
#include "fstream"
#include "sstream"
#include "math.h"
#include "chrono"
#include "random"
#include "iomanip"
#include "eSNN.h"
#include <cstdlib>
#include <ctime>

void LoadDataset(string filename, Dataset * trainingDataset);
void LoadDatasetClasses(string filename, Dataset * trainingDataset);
void PrintDataset(Dataset * d);
vector<Dataset *> GenerateSamples(Dataset * trainingDataset, int N);
void SaveResults(string path, Dataset * d);
double AvgCalculateRecall(Dataset* testDataset, Dataset* trainingDataset);
double AvgCalculatePrecision(Dataset* testDataset, Dataset* trainingDataset);
double Acc(Dataset *testDataset);
void PrintResults(Dataset * d, string fileName);

#endif //PREDICTION_LOADDATA_H
