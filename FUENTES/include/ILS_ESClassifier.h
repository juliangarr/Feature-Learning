#ifndef ILS_ES_CLASSIFIER_H
#define ILS_ES_CLASSIFIER_H

#include "ILSClassifier.h"

#include <tuple>

using namespace std;

class ILS_ESClassifier : public ILSClassifier{        
    public:
        // Constructor
        ILS_ESClassifier(const vector<vector<DataInstance>>& datos, string type);

        // Sobreescritura del método train (llama a ILS_ES)
        tuple<vector<double>, double> train(int fold) override;

        // Método para aplicar ES a una sola solucion
        double ESIndividual(vector<double>& solucionES, int numIteraciones);
};

#endif