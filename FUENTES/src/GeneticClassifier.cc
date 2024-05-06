#include "GeneticClassifier.h"

#include "random.hpp"
#include <tuple>
//#include <Eigen/Dense>

using namespace std;
using Random = effolkronium::random_static;

GeneticClassifier::GeneticClassifier(const vector<vector<DataInstance>>& datos, string type, bool blx_alpha) : RandomToolsClassifier(datos, type) {
    // Inicializar el mejor individuo
    bestIndividual = 0;
    bestFitness = 0;
    bestWeights = vector<double>(numFeatures);

    // Inicializar el número de padres a cero para que las clases hijas lo inicialicen adecuadamente
    numPadres = 0;

    // Inicializar el número esperado de cruces a cero para que las clases hijas lo inicialicen adecuadamente
    numEsperadoCruces = 0;

    // Inicializar las fitnesses
    fitnesses = vector<double>(TAM_POBLACION);

    // Inicializar el booleano de BLX a false   ----------------------------------------------------------------------
    // ---------------------------------------------------------------------------------------------------------------
    // ---------------------------------------------------------------------------------------------------------------
    blx = blx_alpha;
}

tuple<vector<double>, double> GeneticClassifier::train(int fold) {
    return AGG(fold);
}

tuple<vector<double>, double> GeneticClassifier::AGG(int fold) {
    // Generar la población inicial
    vector<vector<double>> population(TAM_POBLACION);
    for(int i = 0; i < TAM_POBLACION; i++) {
        population[i] = generaSolucionInicial();
    }

    // Evaluar la población inicial
    evaluarPoblacion(population);

    // Crear el vector de sucesores
    vector<vector<double>> sucesores(numPadres);

    while(evaluationsDone < MAX_EVALUATIONS){
        // Generación de los padres
        generarPadres(numPadres, population, sucesores);

        // Cruce de los padres
        cruce(sucesores);

        // Mutación para generar los hijos
        mutacion(sucesores);

        // Reemplazo de la población (Y EVALUACIÓN DE LA NUEVA POBLACIÓN DENTRO DE REEMPLAZO)
        reemplazo(population, sucesores);
    }
    // Calculate the best weights
    mejorIndividuo();
    bestWeights = population[bestIndividual];

    // Calculate real fitness (NOT IN TRAINING SET -> TEST SET)
    double realFitness = funcionObjetivo(bestWeights, fold); 

    return make_tuple(bestWeights, realFitness);
}

void GeneticClassifier::evaluarPoblacion(const vector<vector<double>>& population) {
    for(int i = 0; i < TAM_POBLACION; i++) {
        fitnesses[i] = funcionObjetivoLocal(population[i]);
    }
    // Actualizar el mejor individuo de la población
    mejorIndividuo();
}
// void GeneticClassifier::evaluarPoblacion(const vector<vector<double>>& population) {
//     MatrixXd populationMatrix(TAM_POBLACION, numFeatures);

//     // Rellenar la matriz de población
//     for (int i = 0; i < TAM_POBLACION; ++i) {
//         for (int j = 0; j < numFeatures; ++j) {
//             populationMatrix(i, j) = population[i][j];
//         }
//     }

//     // Calcular las funciones de ajuste para toda la población simultáneamente
//     VectorXd fitnessVector = populationMatrix.rowwise().unaryExpr([&](const auto& row) {
//         return funcionObjetivoLocal(row);
//     });

//     // Copiar los resultados de vuelta al vector de funciones de ajuste
//     for (int i = 0; i < TAM_POBLACION; ++i) {
//         fitnesses[i] = fitnessVector(i);
//     }

//     // Actualizar el mejor individuo de la población
//     mejorIndividuo();
// }

void GeneticClassifier::mejorIndividuo() {
    // Encontrar el iterador al elemento máximo en el vector
    auto maxIter = min_element(fitnesses.begin(), fitnesses.end());

    // Calcular el índice del elemento máximo y actualizar el mejor individuo
    bestIndividual = distance(fitnesses.begin(), maxIter);
    bestFitness = *maxIter;
}
// void GeneticClassifier::mejorIndividuo() {
//     // Convertir el vector de funciones de ajuste a una matriz de Eigen
//     VectorXd fitnessVector(TAM_POBLACION);
//     for (int i = 0; i < TAM_POBLACION; ++i) {
//         fitnessVector(i) = fitnesses[i];
//     }

//     // Encontrar el índice del mejor individuo
//     int maxIndex;
//     double maxFitness = fitnessVector.maxCoeff(&maxIndex);

//     // Actualizar el mejor individuo y su ajuste
//     bestIndividual = maxIndex;
//     bestFitness = maxFitness;
// }

vector<double> GeneticClassifier::seleccionTorneo(const vector<vector<double>>& population) {
    vector<double> bestParent;
    double bestFit = __DBL_MIN__;
    int index;
    double fitness;

    for(int i = 0; i < TAM_TORNEO; i++) {
        index = Random::get(0, TAM_POBLACION - 1);
        fitness = fitnesses[index];
        if(fitness > bestFit) {
            bestFit = fitness;
            bestParent = population[index];
        }
    }

    return bestParent;
}

void GeneticClassifier::generarPadres(int numPadres, const vector<vector<double>>& population, vector<vector<double>>& sucesores) {
    // Hacemos torneo tantas veces como número de padres
    for(int i = 0; i < numPadres; i++) {
        sucesores[i] = seleccionTorneo(population);
    }
}

void GeneticClassifier::cruce(vector<vector<double>>& population) {
    // Hacemos Nº Esperado de Cruces tomando las "Nº Esperado de cruces" primeras parejas de padres
    for(int i = 0; i < numEsperadoCruces; i++) {
        vector<double> parent1 = population[i*2];
        vector<double> parent2 = population[i*2 + 1];
        
        if(blx) {
            cruceBLX(parent1, parent2);
        } else {
            cruceAritmetico(parent1, parent2);
        }
    }
}

void GeneticClassifier::cruceBLX(vector<double>& padre1, vector<double>& padre2) {
    // Para cada gen de los padres
    for(int i = 0; i < numFeatures; i++) {
        // Buscamos el máximo y el mínimo de cada gen
        double maxGen = max(padre1[i], padre2[i]);
        double minGen = min(padre1[i], padre2[i]);

        // Calculamos el rango
        double rango = maxGen - minGen;

        // Calculamos el rango de cruce
        double rangoCruce = ALPHA_BLX * rango;

        // Calculamos los nuevos valores de los genes y actualizamos los padres
        padre1[i] = Random::get(minGen - rangoCruce, maxGen + rangoCruce);
        padre2[i] = Random::get(minGen - rangoCruce, maxGen + rangoCruce);
    }
}
// void GeneticClassifier::cruceBLX(VectorXd& padre1, VectorXd& padre2) {
//     VectorXd maxGen = padre1.cwiseMax(padre2);
//     VectorXd minGen = padre1.cwiseMin(padre2);
//     VectorXd rango = maxGen - minGen;
//     VectorXd rangoCruce = ALPHA_BLX * rango;

//     // Generar números aleatorios uniformes entre 0 y 1 para cada elemento del vector
//     VectorXd randomNumbers = VectorXd::Random(padre1.size()).array() * 0.5 + 0.5;

//     // Calcular los nuevos valores de los genes y actualizar los padres
//     padre1 = minGen - rangoCruce.array() * randomNumbers.array();
//     padre2 = maxGen + rangoCruce.array() * randomNumbers.array();
// }

void GeneticClassifier::cruceAritmetico(vector<double>& padre1, vector<double>& padre2) {
    // Calculamos el valor de alpha
    double alpha = Random::get(0.0, 1.0);

    // Calculamos los hijos actualizando los padres
    for(int i = 0; i < numFeatures; i++) {
        double aux = padre1[i];
        padre1[i] = alpha * aux + (1 - alpha) * padre2[i];
        padre2[i] = alpha * padre2[i] + (1 - alpha) * aux;
    }
}


// FUNCIONES DE MUTACIÓN Y REEMPLAZO VACÍAS PARA EVITAR ERROR DE COMPLILACIÓN
void GeneticClassifier::mutacion(vector<vector<double>>& sucesores) {
    // Implementación predeterminada (no hace nada)
}

void GeneticClassifier::reemplazo(vector<vector<double>>& population, vector<vector<double>>& sucesores) {
    // Implementación predeterminada (no hace nada)
}