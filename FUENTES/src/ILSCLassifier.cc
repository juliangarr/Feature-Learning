#include "ILSClassifier.h"

#include "random.hpp"
#include <functional> // Necesario para std::function

using namespace std;
using Random = effolkronium::random_static;

ILSClassifier::ILSClassifier(const vector<vector<DataInstance>>& datos, string type) : RandomToolsClassifier(datos, type) {}

tuple<vector<double>, double> ILSClassifier::train(int fold) {
    return ILS(fold, [this](std::vector<double>& solucion, int iteraciones) {
        return this->busquedaLocalSolucion(solucion, iteraciones);
    });
}

tuple<vector<double>, double> ILSClassifier::ILS(int fold, function<double(vector<double>&, int)> funcionMejora) {
    // Reset the counter of evaluations
    evaluationsDone = 0;
    //int maxEvals = MAX_EVALUATIONS;
    int MAX_ITERATIONS = 20;

    // Establecemos los parámetros del algoritmo
    int ITERS_SUBFUNCION = 750;

    // Generamos la solución inicial
    vector<double> bestSolucion = generaSolucionInicial();

    // Aplicamos la búsqueda local sobre la solución inicial
    double currentFitness = funcionMejora(bestSolucion, ITERS_SUBFUNCION);

    // Inicializamos la mejor solución
    double bestFitness = currentFitness;

    // Inicializamos el número de ejecuciones
    int numEjecuciones = 1;

    // Declaramos la solicion mutada
    vector<double> newSolution;

    while (numEjecuciones < MAX_ITERATIONS /*&& evaluationsDone < maxEvals*/) {
        // Generamos una nueva solución
        newSolution = mutacion(bestSolucion);

        // Aplicamos la búsqueda local sobre la nueva solución
        currentFitness = funcionMejora(newSolution, ITERS_SUBFUNCION);

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

    // Devolvemos la mejor solucion (pesos y fitness)
    return make_tuple(bestSolucion, real_fitness);
}

double ILSClassifier::busquedaLocalSolucion(vector<double>& solucionBL, int numIteraciones){
    // Initialize counters
    int consecutiveNeighbors = 0;
    int maxConsecutiveNeighbors = numFeatures*20;
    int evaluationsBL = 0;
    //int maxEvalsGlobal = MAX_EVALUATIONS;

    // Initialize and evaluate the initial solution
    double fitnessSolucion = funcionObjetivoLocal(solucionBL);
    ++evaluationsBL;
    double bestFitBL = fitnessSolucion;
    
    // Generate the index vector
    vector<int> index(numFeatures);
    for(int i = 0; i < numFeatures; ++i) index[i] = i;

    // Create the vector of neighbors
    vector<double> neighborWeights;
    double neighborFitness;

    // Loop of local search
    while ( /*evaluationsDone < maxEvalsGlobal && */consecutiveNeighbors < maxConsecutiveNeighbors && evaluationsBL < numIteraciones) {
        // Generate a random permutation of index about the features
        Random::shuffle(index);

        // Loop of neighborhood generation
        for (int i = 0; i < numFeatures; ++i){
            if (consecutiveNeighbors < maxConsecutiveNeighbors && /*evaluationsDone < maxEvalsGlobal &&*/ evaluationsBL < numIteraciones) {
                // Generate a neighbor solution
                neighborWeights = solucionBL;
                neighborWeights[index[i]] = mutacionMovNormalBL(solucionBL[index[i]]);
                ++consecutiveNeighbors;

                // Evaluate the neighbor solution
                neighborFitness = funcionObjetivoLocal(neighborWeights);
                ++evaluationsBL;

                // Update the best solution in the BL
                if (neighborFitness > bestFitBL) {
                    solucionBL = neighborWeights;
                    bestFitBL = neighborFitness;
                    consecutiveNeighbors = 0;
                    break;
                }
            }
        }
    }
    return bestFitBL;
}

double ILSClassifier::mutacionMovNormalBL(double peso_pos) const {
    // Create a normal distribution
    normal_distribution<double> normalDistribution(MEDIA, DESVIACION_ESTANDAR);

    // Modify the weight of the feature with a random value (normal distribution)
    peso_pos += Random::get(normalDistribution);

    // Truncate the value to the range [0, 1]
    if (peso_pos < 0.0) {
        return 0.0;
    }
    else if (peso_pos > 1.0) {
        return 1.0;
    }
    else {
        return peso_pos;
    }
}

vector<double> ILSClassifier::mutacion(const vector<double>& solucion) const {
    // Escoger cuantos genes mutar
    int t = round(numFeatures*0.2);

    // Escogemos t características aleatorias (las t primeras)
    vector<int> index(numFeatures);
    for(int i = 0; i < numFeatures; ++i) index[i] = i;
    Random::shuffle(index);

    // Mutamos las características seleccionadas
    vector<double> nuevaSolucion = solucion;
    for (int i = 0; i < t; ++i) {
        nuevaSolucion[index[i]] += Random::get(-0.25, 0.25);

        // Truncamos el valor al rango [0, 1]
        if (nuevaSolucion[index[i]] < 0.0) {
            nuevaSolucion[index[i]] = 0.0;
        }
        else if (nuevaSolucion[index[i]] > 1.0) {
            nuevaSolucion[index[i]] = 1.0;
        }
    }

    return nuevaSolucion;
}