#ifndef BMB_CLASSIFIER_H
#define BMB_CLASSIFIER_H

#include "ILSClassifier.h"

#include <tuple>

using namespace std;

class BMBClassifier : public ILSClassifier{        
    public:
        // Constructor
        BMBClassifier(const vector<vector<DataInstance>>& datos, string type);

        // Sobreescritura del método train (llama a BMB)
        tuple<vector<double>, double> train(int fold) override;

        // Algoritmo de búsqueda multiarranque básica
        tuple<vector<double>, double> BMB(int fold);
};

#endif