#include "ESClassifier.h"

#include "random.hpp"

using namespace std;
using Random = effolkronium::random_static;

ESClassifier::ESClassifier(const vector<vector<DataInstance>>& datos, string type) : RandomToolsClassifier(datos, type) {}

tuple<vector<double>, double> ESClassifier::train(int fold) {
    return ES(fold);
}

tuple<vector<double>, double> ESClassifier::ES(int fold) {
    // Reseteamos el contador de evaluaciones
    evaluationsDone = 0;

    // Definimos los parámetros límite de los bucles del algoritmo
    int maxNeighbors = numFeatures*10;
    double maxExits = 0.1*maxNeighbors;
    int maxEvals = MAX_EVALUATIONS;      // Para trabajar con una variable local (más eficiente)
    int M = maxEvals/maxNeighbors;       // Parámetro para máximo de enfriamientos

    // Generamos la solución inicial
    vector<double> solucion = generaSolucionInicial();

    // Inicializamos la mejor solucion
    vector<double> bestWeights(solucion);
    
    // Evaluamos la solución inicial
    double fitness = funcionObjetivoLocal(solucion);

    // Actualizamos la mejor solución
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
    while( evaluationsDone < maxEvals && numExits != 0 ) {
        // Reseteamos los contadores
        numNeighbors = 0;
        numExits = 0;

        // Bucle (INTERNO)
        while ( evaluationsDone < maxEvals && numNeighbors < maxNeighbors && numExits < maxExits ) {
            // Generamos un vecino
            vecino = solucion;
            mutacionMovNormal(vecino);
            // Actualizamos el contador de vecinos
            ++numNeighbors;

            // Evaluamos el vecino y actualizamos el numero de evaluaciones de forma interna
            fitnessVecino = funcionObjetivoLocal(vecino);

            // Calculamos la diferencia de fitness
            delta_f = fitness - fitnessVecino;

            // Actualizamos la mejor solución
            if (delta_f < 0 || Random::get<double>(0.0, 1.0) <= exp(-delta_f/temp) ) {
                solucion = vecino;
                fitness = fitnessVecino;
                if (fitness > bestFitness) {
                    bestWeights = solucion;
                    bestFitness = fitness;
                }
                // Actualizamos el contador de exitos
                numExits++;
            }
        }
        // Aplicamos el enfriamiento actualizando la temperatura
        temp = temp/(1 + BETA*temp);
    }

    // Calculamos el REAL fitness
    double real_fitness = funcionObjetivo(bestWeights, fold); 

    // Devolvemos la mejor solucion (pesos y fitness)
    return make_tuple(bestWeights, real_fitness);
}