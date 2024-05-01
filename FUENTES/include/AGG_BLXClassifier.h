#ifndef AGG_BLX_CLASSIFIER_H
#define AGG_BLX_CLASSIFIER_H

#include "GeneticClassifier.h"

using namespace std;

class AGG_BLXClassifier : public GeneticClassifier{
    protected:
        static constexpr double PROB_CRUCE = 0.68;

    public:
        // Constructor
        AGG_BLXClassifier(const vector<vector<DataInstance>>& datos, string type);

        // Función de evaluación inicial de la población
        void evaluarPoblacion(const vector<vector<double>>& population) override;

        // Función de selección de padres (depende del tipo de algoritmo genético)
        vector<vector<double>> generarPadres(const vector<vector<double>>& population) override;

        // Función de cruce (depende del tipo de algoritmo genético)
        vector<vector<double>> cruce(const vector<vector<double>>& parents) override;

        // Función de mutación (depende solo de la probabilidad de mutación)
        void mutacion(vector<vector<double>>& population) override;

        // Función de reemplazo (depende del tipo de algoritmo genético)
        vector<vector<double>> reemplazo(const vector<vector<double>>& population, const vector<vector<double>>& hijos) override;
};

#endif