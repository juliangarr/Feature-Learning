#ifndef ES_CLASSIFIER_H
#define ES_CLASSIFIER_H

#include "RandomToolsClassifier.h"

#include <tuple>

using namespace std;

class ESClassifier : public RandomToolsClassifier{        
    public:
        // Constructor
        ESClassifier(const vector<vector<DataInstance>>& datos, string type);

        // Sobreescritura del método train (llama a ES)
        tuple<vector<double>, double> train(int fold) override;

        // Algoritmo de enfriamiento simulado
        tuple<vector<double>, double> ES(int fold);
};

#endif