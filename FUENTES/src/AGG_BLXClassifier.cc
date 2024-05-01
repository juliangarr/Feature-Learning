#include "AGG_BLXClassifier.h"

#include "random.hpp"

using namespace std;
using Random = effolkronium::random_static;

void AGG_BLXClassifier::evaluarPoblacion(const vector<vector<double>>& population) {
    double fitness;
    for(int i = 0; i < TAM_POBLACION; i++) {
        fitness = funcionObjetivoLocal(population[i]);
        if(fitness > bestFitness) {
            // Update the best individual
            bestFitness = fitness;
            bestWeights = population[i];
        }
        if(fitness < worstFitness1) {            
            // Update the worst individual
            worstFitness1 = fitness;
            worstWeights1 = population[i];
        }
    }
}

// vector<vector<double>> AGG_BLXClassifier::gernerarPadres(const vector<vector<double>>& population) {
//     vector<vector<double>> parents(2);
//     vector<double> fitness(TAMAÑO_POBLACION);
//     vector<double> probabilidad(TAMAÑO_POBLACION);
//     double suma = 0.0;

//     // Calculate the fitness of each individual
//     for(int i = 0; i < TAMAÑO_POBLACION; i++) {
//         fitness[i] = funcionObjetivoLocal(population[i]);
//         suma += fitness[i];
//     }

//     // Calculate the probability of each individual
//     for(int i = 0; i < TAMAÑO_POBLACION; i++) {
//         probabilidad[i] = fitness[i] / suma;
//     }

//     // Select the first parent
//     double random = Random::get<double>(0.0, 1.0);
//     double acumulado = 0.0;
//     for(int i = 0; i < TAMAÑO_POBLACION; i++) {
//         acumulado += probabilidad[i];
//         if(acumulado >= random) {
//             parents[0] = population[i];
//             break;
//         }
//     }

//     // Select the second parent
//     random = Random::get<double>(0.0, 1.0);
//     acumulado = 0.0;
//     for(int i = 0; i < TAMAÑO_POBLACION; i++) {
//         acumulado += probabilidad[i];
//         if(acumulado >= random) {
//             parents[1] = population[i];
//             break;
//         }
//     }

//     return parents;
// }

// for(int i = 0; i < TAM_POBLACION; i++) {
//         fitness = funcionObjetivoLocal(population[i]);
//         if(fitness > bestFitness) {
//             // Update the best individual
//             bestFitness = fitness;
//             bestWeights = population[i];
//         }
//         if(fitness < worstFitness1) {
//             // Update the second worst individual
//             worstFitness2 = worstFitness1;
//             worstWeights2 = worstWeights1;
            
//             // Update the worst individual
//             worstFitness1 = fitness;
//             worstWeights1 = population[i];
//         } 
//         else if(fitness < worstFitness2) {
//             // Update the second worst individual
//             worstFitness2 = fitness;
//             worstWeights2 = population[i];
//         }
//     }



