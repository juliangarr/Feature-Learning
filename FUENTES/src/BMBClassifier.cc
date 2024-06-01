#include "BMBClassifier.h"

#include "random.hpp"

using namespace std;
using Random = effolkronium::random_static;

BMBClassifier::BMBClassifier(const vector<vector<DataInstance>>& datos, string type) : ILSClassifier(datos, type) {}

tuple<vector<double>, double> BMBClassifier::train(int fold) {
    return BMB(fold);
}

tuple<vector<double>, double> BMBClassifier::BMB(int fold) {
    // Reset the counter of evaluations
    evaluationsDone = 0;
    
    // Inicializamos el número de ejecuciones
    int numEjecuciones = 0;

    // Establecemos los parámetros del algoritmo
    int ITERS_BL = 750;
    int MAX_ITERATIONS = 20;

    // Generamos la solución para iterar y la mejor solucion
    vector<double> newSolution;
    vector<double> bestSolucion;

    // Declaramos variables para almacenar el fitness
    double currentFitness;
    double bestFitness = -1;

    while (numEjecuciones < MAX_ITERATIONS) {
        // Generamos una nueva solución
        newSolution = generaSolucionInicial();

        // Aplicamos la búsqueda local sobre la nueva solución
        currentFitness = busquedaLocalSolucion(newSolution, ITERS_BL);

        // Actualizamos la mejor solución
        if (currentFitness > bestFitness) {
            bestSolucion = newSolution;
            bestFitness = currentFitness;
        }

        // Incrementamos el número de ejecuciones
        ++numEjecuciones;
    }

    // Calculamos el REAL fitness
    double real_fitness = funcionObjetivo(bestSolucion, fold); 

    printf("Evaluaciones BMB: %d\n", evaluationsDone);

    // Devolvemos la mejor solucion (pesos y fitness)
    return make_tuple(bestSolucion, real_fitness);
}