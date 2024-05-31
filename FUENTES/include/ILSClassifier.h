#ifndef ILS_CLASSIFIER_H
#define ILS_CLASSIFIER_H

#include "RandomToolsClassifier.h"

#include <tuple>
#include <functional> // Necesario para std::function

using namespace std;

class ILSClassifier : public RandomToolsClassifier{        
    public:
        // Constructor
        ILSClassifier(const vector<vector<DataInstance>>& datos, string type);

        // Sobreescritura del método train (llama a ILS)
        tuple<vector<double>, double> train(int fold) override;

        // Algoritmo de ILS
        tuple<vector<double>, double> ILS(int fold, function<double(vector<double>&, int)> funcionMejora);

        // Búsqueda Local sobre una solucion
        double busquedaLocalSolucion(vector<double>& solucionBL, int numIteraciones);

        // Mutación de una característica (específico de BL)
        double mutacionMovNormalBL(double peso_pos) const;

        // Mutación de una solucion significativamente (específico de ILS)
        vector<double> mutacion(const vector<double>& solucion) const;
};

#endif