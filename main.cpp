#include <iostream>
#include "eSNN.h"
#include "LoadData.h"

int main() {

    string Path = "../Datasets/";


    double avAcc = 0;

    for (simTr = 0.05; simTr <= 0.05; simTr += 0.1) {
        for (Bins = 3; Bins <= 3; Bins += 1) {
            for (NIsize = 15; NIsize <= 15; NIsize += 10) {

                Dataset *trainingDataset = new Dataset;
                Dataset *testDataset = new Dataset;

                LoadDataset(Path + "TrainTitleB.csv",
                            trainingDataset);

                LoadDatasetClasses(Path + "TrainTitleBClass.csv",
                                   trainingDataset);

                LoadDataset(Path + "TestTitleB.csv",
                            testDataset);

                LoadDatasetClasses(Path + "TestTitleBClass.csv",
                                   testDataset);

                string resultsFilename = Path + "ResultsMeasures.csv";

                cout << "Loaded" << endl;
                // simTr = 0.15;
                //NIsize = 12;

                mod = 0.95;
                // Bins = 3;

                if (Bins > NIsize) Bins = NIsize;

                cout << simTr << " " << NIsize << " " << Bins << " " << mod << " " << endl;

                //PrintDataset(trainingDataset);


                eSNN *eSNN_net = new eSNN;

                InitializeInputLayerDist(eSNN_net, trainingDataset, testDataset);


                eSNN_Learn(eSNN_net, trainingDataset);
                cout << endl;
                eSNN_Indexing(eSNN_net, testDataset);

                double acc = Acc(testDataset);
                cout << endl << acc << endl;
                avAcc += acc;


                fstream results;
                results.open("../Results/ResultsShortTitleBMesh.csv", fstream::out);
                results << simTr << " " << NIsize << " " << mod << " " << Bins << endl;
                for (int i = 0; i < testDataset->predictedClasses.size(); i++) {
                    for (int j = 0; j < testDataset->predictedClasses[i].size(); j++) {
                        results << testDataset->predictedClasses[i][j] << "," << testDataset->realClasses[i][j];
                    }
                    results << endl;
                }
                results.close();


                ClearStructures(eSNN_net, trainingDataset, testDataset);

            }
        }
    }
}

