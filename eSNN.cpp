//
// Created by Piotr Maciąg on 14/06/2021.
//


#include "eSNN.h"

int NIsize;
double simTr;
double mod;
int K;
int Bins;

void InitializeInputLayerGRFs(eSNN *eSNN_net, Dataset *trainingDataset, Dataset *testDataset) {
    for (int k = 0; k < trainingDataset->att.size(); k++) {

        inputAttribute *InputNeuronsVect = new inputAttribute;
        for (int j = 0; j < NIsize; j++) {
            inputNeuron *newInputNeuron = new inputNeuron{j, double(k), 0};
            InputNeuronsVect->InputNeurons.push_back(newInputNeuron);
        }

        eSNN_net->Attribute.push_back(InputNeuronsVect);

        double max = -2000000;
        double min = 2000000;

        for (int i = 0; i < trainingDataset->inputValues.size(); i++) {
            if (k < trainingDataset->inputValues[i].size()) {
                if (max < trainingDataset->inputValues[i][k]) max = trainingDataset->inputValues[i][k];
                if (min > trainingDataset->inputValues[i][k]) min = trainingDataset->inputValues[i][k];
            }
        }


        eSNN_net->Attribute[k]->I_min = min, eSNN_net->Attribute[k]->I_max = max;
    }

    for (int k = 0; k < eSNN_net->Attribute.size(); k++) {
        for (int j = 0; j < eSNN_net->Attribute[k]->InputNeurons.size(); j++) {
            double mu = eSNN_net->Attribute[k]->I_min + ((2.0 * j - 3.0) / 2.0) * ((eSNN_net->Attribute[k]->I_max - eSNN_net->Attribute[k]->I_min) / (double(NIsize) - 2));
            double sigma =  (((eSNN_net->Attribute[k]->I_max - eSNN_net->Attribute[k]->I_min) / (double(NIsize) - 2)));

            if(sigma == 0.0) sigma = 1.0;

            eSNN_net->Attribute[k]->InputNeurons[j]->mu = mu;
            eSNN_net->Attribute[k]->InputNeurons[j]->sigma = sigma;
        }
    }
}

bool compFiringTime(const inputNeuron &nI1, const inputNeuron &nI2) { //comparator of firing times
    if (nI1.firingTime != nI2.firingTime) {
        return nI1.firingTime < nI2.firingTime;
    } else {
        return nI1.ID < nI2.ID;
    }
}

void CalculateOrderGRFs(eSNN *eSNN_net, Example exmp) {

    int min = (exmp.values.size() < eSNN_net->Attribute.size()) ? exmp.values.size() : eSNN_net->Attribute.size();

    for (int k = 0; k < min; k++) {
        vector<inputNeuron> sortInputNeurons;
        for (int j = 0; j < eSNN_net->Attribute[k]->InputNeurons.size(); j++) {

            double exc = (exp(-0.5 * pow(((exmp.values[k] - eSNN_net->Attribute[k]->InputNeurons[j]->mu) / eSNN_net->Attribute[k]->InputNeurons[j]->sigma), 2)));
            double firingTime = /*floor*/(1 - exc);

            eSNN_net->Attribute[k]->InputNeurons[j]->firingTime = firingTime;
            inputNeuron newIN;
            newIN.firingTime = firingTime;
            newIN.ID = j;
            sortInputNeurons.push_back(newIN);
        }
        sort(sortInputNeurons.begin(), sortInputNeurons.end(), compFiringTime);
        int order = 0;
        for (int j = 0; j < eSNN_net->Attribute[k]->InputNeurons.size(); j++) {
            eSNN_net->Attribute[k]->InputNeurons[sortInputNeurons[j].ID]->order = order;
            order++;
        }
    }
}


void InitializeInputLayerDist(eSNN *eSNN_net, Dataset *trainingDataset, Dataset *testDataset) {
    for (int k = 0; k < trainingDataset->att.size(); k++) {

        inputAttribute *InputNeuronsVect = new inputAttribute;
        for (int j = 0; j < NIsize; j++) {
            inputNeuron *newInputNeuron = new inputNeuron{j, double(k), 0};
            InputNeuronsVect->InputNeurons.push_back(newInputNeuron);
        }

        eSNN_net->Attribute.push_back(InputNeuronsVect);

        double max = -2000000;
        double min = 2000000;

        for (int i = 0; i < trainingDataset->inputValues.size(); i++) {
            if (k < trainingDataset->inputValues[i].size()) {
                if (max < trainingDataset->inputValues[i][k]) max = trainingDataset->inputValues[i][k];
                if (min > trainingDataset->inputValues[i][k]) min = trainingDataset->inputValues[i][k];
            }
        }

        for (int i = 0; i < testDataset->inputValues.size(); i++) {
            if (k < testDataset->inputValues[i].size()) {
                if (max < testDataset->inputValues[i][k]) max = testDataset->inputValues[i][k];
                if (min > testDataset->inputValues[i][k]) min = testDataset->inputValues[i][k];
            }
        }


        eSNN_net->Attribute[k]->I_min = min, eSNN_net->Attribute[k]->I_max = max;

        vector<int> neuronsPerBins;

        for (int i = 0; i < Bins; i++) {
            neuronsPerBins.push_back(1);

        }

        double widthBins = double(eSNN_net->Attribute[k]->I_max - eSNN_net->Attribute[k]->I_min) / double(Bins);

        vector<int> ValuesPerBin;

        for (int i = 0; i < Bins; i++) {
            ValuesPerBin.push_back(0);
        }

        for (int i = 0; i < trainingDataset->inputValues.size(); i++) {
            for (int j = 1; j <= Bins - 1; j++) {
                if (trainingDataset->inputValues[i][k] >= (min + (j - 1) * widthBins) &&
                    trainingDataset->inputValues[i][k] < (min + (j) * widthBins)) {
                    ValuesPerBin[j - 1]++;
                }
            }
            if (trainingDataset->inputValues[i][k] >= (min + (Bins - 1) * widthBins) &&
                trainingDataset->inputValues[i][k] <= max) {
                ValuesPerBin[Bins - 1]++;
            }
        }


        int RestNeurons = NIsize - Bins;
        int r = NIsize - Bins;

        for (int i = 0; i < Bins - 1; i++) {

            neuronsPerBins[i] += floor(
                    (double(ValuesPerBin[i]) / double(trainingDataset->inputValues.size())) * r);
            RestNeurons -= neuronsPerBins[i] + 1;
        }

        neuronsPerBins[Bins - 1] += ceil(
                (double(ValuesPerBin[Bins - 1]) / double(trainingDataset->inputValues.size())) * r);



        for (int i = 0; i < Bins; i++) {
            int sum = 0;
            double widthPerBin = widthBins / neuronsPerBins[i];

            if (i > 0) sum += neuronsPerBins[i - 1];
            for (int j = 0; j < neuronsPerBins[i]; j++) {

                double mu = (min + (i) * widthBins) + (j + 1 - 0.5) * widthPerBin;

                eSNN_net->Attribute[k]->InputNeurons[sum + j]->mu = mu;

            }
        }

        eSNN_net->Attribute[k]->width =
                double(eSNN_net->Attribute[k]->I_max - eSNN_net->Attribute[k]->I_min) / double(NIsize);
    }
}

void CalculateOrderDist(eSNN *eSNN_net, Example exmp) {

    int min = (exmp.values.size() < eSNN_net->Attribute.size()) ? exmp.values.size() : eSNN_net->Attribute.size();


    for (int k = 0; k < min; k++) {
        int j;

        if (exmp.values[k] > eSNN_net->Attribute[k]->I_min && exmp.values[k] < eSNN_net->Attribute[k]->I_max &&
            eSNN_net->Attribute[k]->width != 0) {

            // j = floor((exmp.values[k] - eSNN_net->Attribute[k]->I_min) / eSNN_net->Attribute[k]->width) + 1;

            double minDist = abs(exmp.values[k] - eSNN_net->Attribute[k]->InputNeurons[0]->mu);
            j = 1;
            for (int i = 1; i < eSNN_net->Attribute[k]->InputNeurons.size(); i++) {
                if (minDist > abs(exmp.values[k] - eSNN_net->Attribute[k]->InputNeurons[i]->mu)) {
                   j = i + 1;
                   minDist = abs(exmp.values[k] - eSNN_net->Attribute[k]->InputNeurons[i]->mu);
                }
            }
        } else if (exmp.values[k] >= eSNN_net->Attribute[k]->I_max) {
            j = NIsize;
        } else if (exmp.values[k] <= eSNN_net->Attribute[k]->I_min) {
            j = 1;
        }

        eSNN_net->Attribute[k]->InputNeurons[j - 1]->rank = 0;

        int l, p;

         l = j-1;
         p = j + 1;

        int rank = 0;
        double distL = 0;
        double distP = 0;



        while(l >= 1 || p <= NIsize)
        {
            if(l >= 1) distL = abs(eSNN_net->Attribute[k]->InputNeurons[l-1]->mu - exmp.values[k]);
            if(p <= NIsize) distP = abs(eSNN_net->Attribute[k]->InputNeurons[p-1]->mu - exmp.values[k]);


            if(l < 1 && p <= NIsize)
            {
                rank++;
                eSNN_net->Attribute[k]->InputNeurons[p-1]->rank = rank;
                p++;
            }
            else if(l>= 1 && p > NIsize)
            {
                rank++;
                eSNN_net->Attribute[k]->InputNeurons[l-1]->rank = rank;
                l--;
            }else if (l >= 1 && p <= NIsize) {
                if (distL < distP) {
                    rank++;
                    eSNN_net->Attribute[k]->InputNeurons[l-1]->rank = rank;
                    l--;
                } else {
                    rank++;
                    eSNN_net->Attribute[k]->InputNeurons[p-1]->rank = rank;
                    p++;
                }
            }
        }

        for (int j = 0; j < eSNN_net->Attribute[k]->InputNeurons.size(); j++) {

            eSNN_net->Attribute[k]->InputNeurons[j]->order = eSNN_net->Attribute[k]->InputNeurons[j]->rank ;

        }
    }
}


void InitializeInputLayer(eSNN *eSNN_net, Dataset *trainingDataset, Dataset *testDataset) {
    for (int k = 0; k < trainingDataset->att.size(); k++) {

        inputAttribute *InputNeuronsVect = new inputAttribute;
        for (int j = 0; j < NIsize; j++) {
            inputNeuron *newInputNeuron = new inputNeuron{j, double(k), 0};
            InputNeuronsVect->InputNeurons.push_back(newInputNeuron);
        }

        eSNN_net->Attribute.push_back(InputNeuronsVect);

        double max = -2000000;
        double min = 2000000;

        for (int i = 0; i < trainingDataset->inputValues.size(); i++) {
            if (k < trainingDataset->inputValues[i].size()) {
                if (max < trainingDataset->inputValues[i][k]) max = trainingDataset->inputValues[i][k];
                if (min > trainingDataset->inputValues[i][k]) min = trainingDataset->inputValues[i][k];
            }
        }
        eSNN_net->Attribute[k]->I_min = min, eSNN_net->Attribute[k]->I_max = max;

    }
}

int countt = 0;

void CalculateOrder(eSNN *eSNN_net, Example exmp) {
    double width;

    for (int k = 0; k < eSNN_net->Attribute.size(); k++) {

        for (int j = 0; j < eSNN_net->Attribute[k]->InputNeurons.size(); j++) {
              eSNN_net->Attribute[k]->InputNeurons[j]->order = -1;
        }

        width = double(eSNN_net->Attribute[k]->I_max - eSNN_net->Attribute[k]->I_min) / double(NIsize);
        eSNN_net->Attribute[k]->width = width;

        for (int j = 0; j < eSNN_net->Attribute[k]->InputNeurons.size(); j++) {
            double mu = eSNN_net->Attribute[k]->I_min + (j + 1 - 0.5) * width;
            eSNN_net->Attribute[k]->InputNeurons[j]->mu = mu;
        }
    }


    int min = (exmp.values.size() < eSNN_net->Attribute.size()) ? exmp.values.size() : eSNN_net->Attribute.size();


    for (int k = 0; k < min; k++) {
        int j;
        if (exmp.values[k] > eSNN_net->Attribute[k]->I_min && exmp.values[k] < eSNN_net->Attribute[k]->I_max &&
            eSNN_net->Attribute[k]->width != 0) {
            j = floor((exmp.values[k] - eSNN_net->Attribute[k]->I_min) / eSNN_net->Attribute[k]->width) + 1;
            //cout << "hit" << endl;
            // if(j == 5) {cout << j <<  "  "<< ++countt <<  " " << k << endl;}
        } else if (exmp.values[k] >= eSNN_net->Attribute[k]->I_max) {
            j = NIsize;
        } else if (exmp.values[k] <= eSNN_net->Attribute[k]->I_min) {
            j = 1;
        }


        int l;
        if (j - 1 < NIsize - j) { l = j - 1; } else { l = NIsize - j; }


        eSNN_net->Attribute[k]->InputNeurons[j - 1]->rank = 0;


        if (exmp.values[k] < eSNN_net->Attribute[k]->InputNeurons[j - 1]->mu) {

            for (int n = 1; n <= l; n++) {
                eSNN_net->Attribute[k]->InputNeurons[j - n - 1]->rank = 2 * n - 1;
                eSNN_net->Attribute[k]->InputNeurons[j + n - 1]->rank = 2 * n;
            }
            for (int n = 1; n <= j - 1 - l; n++) //n is k in algorithms
            {
                eSNN_net->Attribute[k]->InputNeurons[j - l - n - 1]->rank = 2 * l  + n; //2*l - 1 + n
            }
            for (int n = 1; n <= NIsize - j - l; n++) //n is k in algorithms
            {
                eSNN_net->Attribute[k]->InputNeurons[j + l + n - 1]->rank = 2 * l + n;
            }
        } else {


            for (int n = 1; n <= l; n++) {
                eSNN_net->Attribute[k]->InputNeurons[j - n - 1]->rank = 2 * n;
                eSNN_net->Attribute[k]->InputNeurons[j + n - 1]->rank = 2 * n - 1;
            }
            for (int n = 1; n <= j - 1 - l; n++) //n is k in algorithms
            {
                eSNN_net->Attribute[k]->InputNeurons[j - l - n - 1]->rank = 2 * l + n;
            }
            for (int n = 1; n <= NIsize - j - l; n++) //n is k in algorithms
            {
                eSNN_net->Attribute[k]->InputNeurons[j + l + n - 1]->rank = 2 * l  + n; //2*l  - 1 + n
            }

        }


        for (int j = 0; j < eSNN_net->Attribute[k]->InputNeurons.size(); j++) {
            eSNN_net->Attribute[k]->InputNeurons[j]->order = eSNN_net->Attribute[k]->InputNeurons[j]->rank;
        }

    }
}


void InitializeNeuron(eSNN *eSNN_net, neuron *n_c, int x, double idx, Example exmp) { //Initalize new neron n_i

    for (int l = 0; l < eSNN_net->Attribute.size(); l++) {
        vector<double> vec;
        n_c->s_weights.push_back(vec);
        for (int j = 0; j < eSNN_net->Attribute[l]->InputNeurons.size(); j++) {
            n_c->s_weights[l].push_back(0.0);
        }
    }

    for (int l = 0; l < exmp.values.size(); l++) {
        for (int j = 0; j < eSNN_net->Attribute[l]->InputNeurons.size(); j++) {
            n_c->s_weights[l][j] = pow(mod, eSNN_net->Attribute[l]->InputNeurons[j]->order);
        }
    }

    n_c->M = 1;
    n_c->ID = idx;
    n_c->cl = x;
}

double
CalculateDistance(const vector<vector<double>> &w1,
                  const vector<vector<double>> &w2) { //calculate distance between two weights vectors
    long double diffSq = 0.0;

    for (int k = 0; k < w1.size(); k++) {
        for (int j = 0; j < w1[k].size(); j++) {
            diffSq += pow(w1[k][j] - w2[k][j], 2);
        }
    }

    return sqrt(diffSq);
}

neuron *
FindMostSimilar(vector<neuron *> NeuClass, neuron *n_c) { //find mos similar neurons in terms of synaptic weights

    double minDist = CalculateDistance(n_c->s_weights, NeuClass[0]->s_weights);
    double minIdx = 0;

    if (NeuClass.size() > 1) {
        for (int i = 1; i < NeuClass.size(); i++) {
            double dist = CalculateDistance(n_c->s_weights, NeuClass[i]->s_weights);
            if (dist < minDist) {
                minDist = dist;
                minIdx = i;
            }
        }
    }
    return NeuClass[minIdx];
}

void UpdateRepository(eSNN *eSNN_net, neuron *n_c, double Dub) { //Update neuron n_s in output repository


    neuron *n_s;

    if (eSNN_net->OutputNeurons[n_c->cl].size() > 0) {
        n_s = FindMostSimilar(eSNN_net->OutputNeurons[n_c->cl], n_c);
    }


    if (eSNN_net->OutputNeurons[n_c->cl].size() > 0 &&
        CalculateDistance(n_c->s_weights, n_s->s_weights) <= simTr * Dub) {

        for (int k = 0; k < n_s->s_weights.size(); k++) {
            for (int j = 0; j < n_s->s_weights[k].size(); j++) {
                n_s->s_weights[k][j] = (n_c->s_weights[k][j] + n_s->s_weights[k][j] * n_s->M) / (n_s->M + 1);
            }
        }

        n_s->M += 1;
        delete n_c;
    } else {
        eSNN_net->OutputNeurons[n_c->cl].push_back(n_c);
    }

}

double CalculateUpperBound(eSNN *eSNN_net) {
    long double sqrtDiff = 0.0;
    for (int i = 0; i < eSNN_net->Attribute.size(); i++) {
        for (int j = 0; j < eSNN_net->Attribute[i]->InputNeurons.size(); j++) {
            sqrtDiff += pow(pow(mod, j) - pow(mod, NIsize - j - 1), 2);
        }
    }

    return (sqrt(sqrtDiff));
}


void eSNN_Learn(eSNN *eSNN_net, Dataset *trainingDataset) { //main eSNN procedure

    double Dub = CalculateUpperBound(eSNN_net);
    eSNN_net->OutputNeurons.resize(trainingDataset->maxClass);

    for (int i = 0; i < trainingDataset->inputValues.size(); i++) {

       // if (i % 1000 == 0)
         //   cout << "Training document processing: " << i << endl;

        if (i % 1000 == 0)
            cout << "#" << flush;

        Example exmp;

        for (int j = 0; j < trainingDataset->inputValues[i].size(); j++) {
            exmp.values.push_back(trainingDataset->inputValues[i][j]);
        }

        CalculateOrderDist(eSNN_net, exmp);

        for (int j = 0; j < trainingDataset->realClasses[i].size(); j++) {
            neuron *n_c = new neuron;
            InitializeNeuron(eSNN_net, n_c, trainingDataset->realClasses[i][j], i, exmp);
            UpdateRepository(eSNN_net, n_c, Dub);

        }
    }
}

struct PSPcount {
    int i = 0;
    int j = 0;
    double mPSP = 0;
    int c;
};

bool comparison(const PSPcount &i1, const PSPcount &i2) {
    if (i1.mPSP != i2.mPSP) {
        return i1.mPSP > i2.mPSP;
    }
    else
    {
        //cout << "eq" << endl;
        return i1.i < i2.i;
    }
}


int mostFrequent(vector<int> arr)
{
    // Sort the array
    sort(arr.begin(), arr.end());

    // find the max frequency using linear traversal
    int max_count = 1, res = arr[0], curr_count = 1;
    for (int i = 1; i < arr.size(); i++) {
        if (arr[i] == arr[i - 1])
            curr_count++;
        else {
            if (curr_count > max_count) {
                max_count = curr_count;
                res = arr[i - 1];
            }
            curr_count = 1;
        }
    }

    // If last element is most frequent
    if (curr_count > max_count)
    {
        max_count = curr_count;
        res = arr[arr.size() - 1];
    }

    return res;
}


vector<int> PredictValue(eSNN *eSNN_net) {
    vector<int> PredictedVals;

    vector<double> maxPSP;
    vector<PSPcount> cP;

    for (int i = 0; i < eSNN_net->OutputNeurons.size(); i++) {
        for (int j = 0; j < eSNN_net->OutputNeurons[i].size(); j++) {
            eSNN_net->OutputNeurons[i][j]->PSP = 0;
        }
    }

    vector<PSPcount> psps;

    for (int i = 0; i < eSNN_net->OutputNeurons.size(); i++) {
        for (int k = 0; k < eSNN_net->OutputNeurons[i].size(); k++) {
            for (int l = 0; l < eSNN_net->Attribute.size(); l++) {
                for (int j = 0; j < eSNN_net->Attribute[l]->InputNeurons.size(); j++) {

                    eSNN_net->OutputNeurons[i][k]->PSP += eSNN_net->OutputNeurons[i][k]->s_weights[l][j] *
                                                          pow(mod, eSNN_net->Attribute[l]->InputNeurons[j]->order);

                    //if (eSNN_net->OutputNeurons[i][k]->PSP >= C * eSNN_net->Attribute.size()*(1 - pow(mod, 2 * NIsize)) / (1 - pow(mod, 2))) {
                    //  PredictedVals.push_back(eSNN_net->OutputNeurons[i][k]->cl);
                    //double psp = eSNN_net->Attribute.size()*(1 - pow(mod, 2 * NIsize))/(1 - pow(mod, 2));
                   // cout << eSNN_net->OutputNeurons[i][k]->PSP << endl;
                    // cout << i << " " << k << endl;
                    // goto label;
                    //   }
                }
            }
//            PSPcount p;
//            p.mPSP = eSNN_net->OutputNeurons[i][k]->PSP;
//            p.i = i;
//            p.j = k;
//            p.c = eSNN_net->OutputNeurons[i][k]->cl;
//            psps.push_back(p);
        }
        //label:;
    }

    int count = -1;
    int cc = -1;
    double mPSP = 0;

    for (int i = 0; i < eSNN_net->OutputNeurons.size(); i++) {
        // PSPcount p;
        //cout << "size " << eSNN_net->OutputNeurons[i].size() << endl;
        for (int k = 0; k < eSNN_net->OutputNeurons[i].size(); k++) {
            if (mPSP < eSNN_net->OutputNeurons[i][k]->PSP) {
                count = k;
                cc = i;
                mPSP = eSNN_net->OutputNeurons[i][k]->PSP;
                //  cout << k << " " <<  i  << " " << mPSP << endl;
                //
            }
        }
        //  p.count = count;
        // p.mPSP = mPSP;
        // p.term = i;
        //PredictedVals.push_back(eSNN_net->OutputNeurons[i][count]->cl);
        //cP.push_back(p);
    }


    //for(int i = 0; i < 4; i++)
    {
        //PredictedVals.push_back(eSNN_net->OutputNeurons[cP[i].term][cP[i].count]->cl);
    }

    //cout << "c" << endl;
    //sort(cP.begin(), cP.end(), comparison);
//    sort(psps.begin(), psps.end(), comparison);
//
//    vector<int> classes;
//
//    for(int i = 0; i < K; i++)
//    {
//        classes.push_back(psps[i].c);
//        cout << "c " << psps[i].c << " " << flush;
//    }
//
//    cout << endl;
//    //cout << "cc"<< endl;
//    int mFreq = mostFrequent(classes);

    //cout << "cc2"<< endl;

    for (int i = 0; i < 1; i++) {
        //PredictedVals.push_back(eSNN_net->OutputNeurons[cP[i].term][cP[i].count]->cl);
        PredictedVals.push_back(eSNN_net->OutputNeurons[cc][count]->cl);
        //PredictedVals.push_back(mFreq);
    }

    //for(int i = 0; i < 4; i++)
    {
        //PredictedVals.push_back(eSNN_net->OutputNeurons[cP[i].term][cP[i].count]->cl);
    }

    return PredictedVals;
}

double eSNN_Indexing(eSNN *eSNN_net, Dataset *testDataset) {
    double acc = 0;
    int count = 0;
    for (int i = 0; i < testDataset->inputValues.size(); i++) {
        Example exmp;

        //if (i % 20 == 0)
          //    xcout << "Testing document processing: " << i << endl;
       // if (i % 100 == 0)
         //   cout << "#" << flush;


        for (int j = 0; j < testDataset->inputValues[i].size(); j++) {
            exmp.values.push_back(testDataset->inputValues[i][j]);
        }

        CalculateOrderDist(eSNN_net, exmp);

        vector<int> vect;
        testDataset->predictedClasses.push_back(vect);
        testDataset->predictedClasses[i] = PredictValue(eSNN_net);

        if (i % 100 == 0)
            cout << "#" << flush;
        double P = 0;
        for (int ii = 0; ii < testDataset->predictedClasses.size(); ii++) {
            for (int j = 0; j < testDataset->predictedClasses[ii].size(); j++) {

                if (testDataset->realClasses[ii][j] == testDataset->predictedClasses[ii][j]) P += 1;
            }
        }
        //cout << testDataset->realClasses[i][0] << " " << testDataset->predictedClasses[i][0] << endl;
        count++;
        if (i % 100 == 0)
        {cout << ceil((P / double(count) )*100.0)/100.0<< flush;}


    }
}

void
ClearStructures(eSNN *eSNN_Nets, Dataset *trainingDataset, Dataset *testDataset) {

    for (int i = 0; i < eSNN_Nets->OutputNeurons.size(); i++) {
        for (int j = 0; j < eSNN_Nets->OutputNeurons[i].size(); j++) {
            delete eSNN_Nets->OutputNeurons[i][j];
        }
    }

    for (int j = 0; j < eSNN_Nets->Attribute.size(); j++) {
        for (int k = 0; k < eSNN_Nets->Attribute[j]->InputNeurons.size(); k++) {
            delete eSNN_Nets->Attribute[j]->InputNeurons[k];
        }

        delete eSNN_Nets->Attribute[j];
    }

    delete eSNN_Nets;

    delete trainingDataset;
    delete testDataset;
}
