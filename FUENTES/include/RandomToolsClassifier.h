#ifndef RANDOM_TOOLS_CLASSIFIER_H
#define RANDOM_TOOLS_CLASSIFIER_H

#include "Classifier.h"
#include <cmath>

using namespace std;

class RandomToolsClassifier : public Classifier{
    protected:
        static constexpr int MAX_EVALUATIONS = 15000;
        static constexpr double MEDIA = 0.0;
        static constexpr double VARIANZA = 0.3;
        const double DESVIACION_ESTANDAR = sqrt(VARIANZA);

        // Contador de evaluaciones de la función objetivo
        int evaluationsDone;

    public:
        // Constructor
        RandomToolsClassifier(const vector<vector<DataInstance>>& datos, string type);

        // Genera una solución inicial (de pesos) aleatoria
        vector<double> generaSolucionInicial() const;

        // Función objetivo en el TRAIN (con LEAVE-ONE-OUT)
        double funcionObjetivoLocal(const vector<double>& weights);

        // Calcular classification rate con LEAVE-ONE-OUT
        double calculateClassRateInTrain(const vector<double>& weights) const;

        // Operador de mutación (genera vecino)
        double mutacionMovNormal(double peso_pos) const;
        //vector<double> generaVecino(const vector<double>& weights, int index) const;
};

#endif
