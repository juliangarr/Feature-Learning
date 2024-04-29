#ifndef GREEDY_CLASSIFIER_H
#define GREEDY_CLASSIFIER_H

#include "Classifier.h"

#include <tuple>

using namespace std;

class GreedyClassifier : public Classifier{
    public:
        GreedyClassifier(const vector<vector<DataInstance>>& datos, string type);
        
        tuple<vector<double>, double> train(int fold) override;

        tuple<vector<double>, double> greedy(int fold);
};

#endif
