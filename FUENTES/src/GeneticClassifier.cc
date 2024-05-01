#include "GeneticClassifier.h"

#include "random.hpp"

using namespace std;
using Random = effolkronium::random_static;

GeneticClassifier::GeneticClassifier(const vector<vector<DataInstance>>& datos, string type) : RandomToolsClassifier(datos, type) {
    // Inicializar mejor y peor fitness-individuo
    bestFitness = __DBL_MIN__;
    worstFitness1 = __DBL_MAX__;

    // Inicializar el número de mutaciones
    num_ind_mutar = PROB_MUT * TAM_POBLACION;
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

    while(evaluationsDone < MAX_EVALUATIONS){
        // Copia de los padres
        vector<vector<double>> parents(population);

        // Selección de los padres
        population = generarPadres(population);

        // Cruce de los padres
        vector<vector<double>> hijos = cruce(parents);

        // Mutación de los hijos
        mutacion(hijos);

        // Reemplazo de la población
        population = reemplazo(population, hijos);

        // Evaluación de la población
        evaluarPoblacion(population);
    }

    // Calculate real fitness (NOT IN TRAINING SET -> TEST SET)
    double fitness = funcionObjetivo(bestWeights, fold); 

    return make_tuple(bestWeights, fitness);
}

vector<double> GeneticClassifier::seleccionTorneo(const vector<vector<double>>& population) {
    vector<double> bestParent;
    double bestFitness = __DBL_MIN__;
    int index;
    double fitness;

    for(int i = 0; i < TAM_TORNEO; i++) {
        index = Random::get(0, TAM_POBLACION - 1);
        fitness = funcionObjetivoLocal(population[index]);
        if(fitness > bestFitness) {
            bestFitness = fitness;
            bestParent = population[index];
        }
    }

    return bestParent;
}

void GeneticClassifier::mutacion(vector<vector<double>>& population_to_mutate){
    int index_pob;
    int index_gen;
    for(int i = 0; i < num_ind_mutar; i++) {
        index_pob = Random::get(0, TAM_POBLACION - 1);
        index_gen = Random::get(0, numFeatures - 1);
        population_to_mutate[index_pob][index_gen] = mutacionMovNormal(population_to_mutate[index_pob][index_gen]);
    }
}
