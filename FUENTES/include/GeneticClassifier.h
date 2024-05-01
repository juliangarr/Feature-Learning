#ifndef GENETIC_CLASSIFIER_H
#define GENETIC_CLASSIFIER_H

#include "RandomToolsClassifier.h"

using namespace std;

class GeneticClassifier : public RandomToolsClassifier{
    protected:
        static constexpr int TAM_POBLACION = 50;
        static constexpr double PROB_MUT = 0.08;     // Prob. de mutación por individuo y también de genes por individuo
        static constexpr int TAM_TORNEO = 3;

        vector<double> bestWeights;
        vector<double> worstWeights1;
        double bestFitness;
        double worstFitness1;

        int num_ind_mutar;

    public:
        // Constructor
        GeneticClassifier(const vector<vector<DataInstance>>& datos, string type);

        // Sobreescritura del método train (llama a AGG)
        tuple<vector<double>, double> train(int fold) override;

        // Algoritmo genético generacional
        tuple<vector<double>, double> AGG(int fold);

        // Función de evaluación inicial de la población
        virtual void evaluarPoblacion(const vector<vector<double>>& population);

        // Función de selección de padres (depende del tipo de algoritmo genético)
        virtual vector<vector<double>> generarPadres(const vector<vector<double>>& population);

        // Selección por torneo (devuelve el mejor de 3 individuos aleatorios de la población actual) 
        vector<double> seleccionTorneo(const vector<vector<double>>& population);

        // Función de cruce (depende del tipo de algoritmo genético)
        virtual vector<vector<double>> cruce(const vector<vector<double>>& parents);

        // Función de mutación (depende solo de la probabilidad de mutación)
        virtual void mutacion(vector<vector<double>>& population);

        // Función de reemplazo (depende del tipo de algoritmo genético)
        virtual vector<vector<double>> reemplazo(const vector<vector<double>>& population, const vector<vector<double>>& hijos);
};

#endif
