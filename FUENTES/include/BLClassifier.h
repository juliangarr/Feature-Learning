#ifndef BL_CLASSIFIER_H
#define BL_CLASSIFIER_H

#include "RandomToolsClassifier.h"

#include <tuple>

using namespace std;

class BLClassifier : public RandomToolsClassifier{
    protected:
        int MAX_NEIGHBORS;
    public:
        // Constructor
        BLClassifier(const vector<vector<DataInstance>>& datos, string type);

        // Sobreescritura del método train (llama a busquedaLocal)
        tuple<vector<double>, double> train(int fold) override;

        // Algoritmo de búsqueda local
        tuple<vector<double>, double> busquedaLocal(int fold);
};

#endif
