//
// Created by Piotr Maciąg on 14/06/2021.
//

#ifndef PREDICTION_ESNN_H
#define PREDICTION_ESNN_H

#include "iostream"
#include "algorithm"
#include "vector"
#include "fstream"
#include "sstream"
#include "math.h"
#include "chrono"
#include "random"
#include "iomanip"

using namespace std;

struct neuron
{
    int ID;
    vector<vector<double>> s_weights;
    int cl;
    double M;
    double PSP;
    bool fire = true;
}; //output neuron structure

struct inputNeuron
{
    int ID;
    double firingTime;
    int order;
    double mu;
    int rank;
    double sigma;
}; //input neuron structure

struct inputAttribute
{
    vector<inputNeuron* > InputNeurons;
    int type = 0; //real 0, ordinal 1, nominal 2
    double I_max, I_min;
    double width;
};

struct eSNN
{
    vector<vector<neuron *>> OutputNeurons;
    vector<inputAttribute *> Attribute;

    int CNO_size = 0;
};

struct Attribute
{
    int type;
    vector<double> values;
};

struct Dataset
{
    vector<Attribute> att;
    vector<vector<double>> inputValues;
    vector<vector<int>> realClasses;
    vector<vector<int>> predictedClasses;
    int maxClass = -1;
    vector<int> Terms;
};

struct Example
{
    vector<double> values;
    vector<int> attributes;
};


extern int NIsize;
extern double simTr;
extern double mod;
extern int K;
extern int Bins;

extern Dataset TrainingDataset;
//extern vector<vector<double>> TrainingTargetValues;

extern vector<vector<double>> TestDataset;
//extern vector<vector<double>> TrainingTestValues;

void eSNN_Learn(eSNN * eSNN_net, Dataset * trainingDataset);
double eSNN_Indexing(eSNN *eSNN_net, Dataset * testDataset);
void InitializeInputLayer(eSNN *eSNN_net, Dataset * trainingDataset, Dataset * testDataset);
void ClearStructures(eSNN * eSNN_Nets, Dataset * trainingDataset, Dataset * testDataset);
void InitializeInputLayerDist(eSNN *eSNN_net, Dataset *trainingDataset, Dataset *testDataset);
void InitializeInputLayerGRFs(eSNN *eSNN_net, Dataset *trainingDataset, Dataset *testDataset);

#endif //PREDICTION_ESNN_H
