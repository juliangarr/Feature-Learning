#include "ILS_ESClassifier.h"

#include "random.hpp"

using namespace std;
using Random = effolkronium::random_static;

ILS_ESClassifier::ILS_ESClassifier(const vector<vector<DataInstance>>& datos, string type) : ILSClassifier(datos, type) {}

tuple<vector<double>, double> ILS_ESClassifier::train(int fold) {
    return ILS(fold, [this](std::vector<double>& solucion, int iteraciones) {
        return this->ESIndividual(solucion, iteraciones);
    });
}

double ILS_ESClassifier::ESIndividual(vector<double>& solucionES, int numIteraciones) {
    // Definimos los parámetros límite de los bucles del algoritmo
    int maxNeighbors = numFeatures*10;
    double maxExits = 0.1*maxNeighbors;
    int maxEvals = numIteraciones;
    int numEvals = 0;
    int M = maxEvals/maxNeighbors;       // Parámetro de enfriamiento

    // La solución inicial es la que se pasa por parámetro
    
    // Evaluamos la solución inicial
    double fitness = funcionObjetivoLocal(solucionES);
    numEvals++;

    // inicializamos y actualizamos la mejor solución
    vector<double> bestSolucion(solucionES);
    double bestFitness = fitness;

    // Inicializamos la temperatura siguiendo la fórmula
    double temp_0 = (0.1*fitness)/(-log(0.3));

    // Inicializamos la temperatura final
    double temp_f = 0.001;

    // Calculamos el valor de BETA
    const double BETA = (temp_0 - temp_f)/(M*temp_0*temp_f);
    
    // Declaramos las variables auxiliares como contadores o valores cambiantes:
    int numNeighbors = -1;  // contador de vecinos generados
    int numExits = -1;      // contador de exitos
    double temp = temp_0;   // Inicializamos la temperatura actual
    double fitnessVecino;   // Fitness del vecino
    double delta_f;         // Diferencia de fitness
    vector<double> vecino;  // Vecino generado

    // Bucle principal del algoritmo (EXTERNO)
    while( numEvals < maxEvals && numExits != 0 ) {
        // Reseteamos los contadores
        numNeighbors = 0;
        numExits = 0;

        // Bucle (INTERNO)
        while ( evaluationsDone < maxEvals && numNeighbors < maxNeighbors && numExits < maxExits ) {
            // Generamos un vecino
            vecino = solucionES;
            mutacionMovNormal(vecino);
            // Actualizamos el contador de vecinos
            ++numNeighbors;

            // Evaluamos el vecino y actualizamos el numero de evaluaciones de forma interna
            fitnessVecino = funcionObjetivoLocal(vecino);
            numEvals++;

            // Calculamos la diferencia de fitness
            delta_f = fitness - fitnessVecino;

            // Actualizamos la mejor solución
            if (delta_f < 0 || Random::get<double>(0.0, 1.0) <= exp(-delta_f/temp) ) {
                solucionES = vecino;
                fitness = fitnessVecino;
                if (fitness > bestFitness) {
                    bestSolucion = solucionES;
                    bestFitness = fitness;
                }
                // Actualizamos el contador de exitos
                numExits++;
            }
        }
        // Aplicamos el enfriamiento actualizando la temperatura
        temp = temp/(1 + BETA*temp);
    }
    solucionES = bestSolucion;

    return bestFitness;
}