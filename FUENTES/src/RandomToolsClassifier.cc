#include "RandomToolsClassifier.h"

#include "random.hpp"

using namespace std;
using Random = effolkronium::random_static;

RandomToolsClassifier::RandomToolsClassifier(const vector<vector<DataInstance>>& datos, string type) : Classifier(datos, type) {
    evaluationsDone = 0;
}

vector<double> RandomToolsClassifier::generaSolucionInicial() const{
    vector<double> weights(numFeatures);
    
    for (int i = 0; i < numFeatures; i++) {
        weights[i] = Random::get<double>(0.0, 1.0); // Random::get<double>(min, max) usa ya una distribución uniforme
    }

    return weights;
}

// Classification rate with LEAVE-ONE-OUT
double RandomToolsClassifier::calculateClassRateInTrain(const vector<double>& weights) const{
    double numCorrects = 0.0;
    int numTrainingInstances = trainingSet.size();

    for (int i = 0; i < numTrainingInstances; ++i) {
        double minDistance = numeric_limits<double>::max();
        string nearestNeighborClass;
        for (int j = 0; j < numTrainingInstances; ++j) {
            if (i != j) {
                double distance = weightedEuclideanDistance(trainingSet[i].features, trainingSet[j].features, weights);
                if (distance < minDistance && distance > 0.0) {
                    minDistance = distance;
                    nearestNeighborClass = trainingSet[j].classLabel;
                }
            }
        }
        if (nearestNeighborClass == trainingSet[i].classLabel) {
            numCorrects++;
        }
    }
    return ( (100.0 * numCorrects) / numTrainingInstances );
}

double RandomToolsClassifier::funcionObjetivoLocal(const vector<double>& weights) {
    // Update the num of evaluations
    evaluationsDone++;

    return ( ALPHA * calculateClassRateInTrain(weights) + (1.0 - ALPHA) * calculateReductionRate(weights) );
}

double RandomToolsClassifier::mutacionMovNormal(double peso_pos) const {
    // Create a normal distribution
    normal_distribution<double> distribution(MEDIA, DESVIACION_ESTANDAR);

    // Modify the weight of the feature with a random value (normal distribution)
    peso_pos += Random::get(distribution);

    // Truncate the value to the range [0, 1]
    if (peso_pos < 0.0) {
        peso_pos = 0.0;
    }
    else if (peso_pos > 1.0) {
        peso_pos = 1.0;
    }

    return peso_pos;
}
// vector<double> BLClassifier::generaVecino(const vector<double>& weights, int index) const {
//     vector<double> newWeights = weights;

//     // Create a normal distribution
//     normal_distribution<double> distribution(MEDIA, sqrt(VARIANZA));

//     // Modify the weight of the feature with a random value (normal distribution)
//     newWeights[index] += Random::get(distribution);

//     // Truncate the value to the range [0, 1]
//     if (newWeights[index] < 0.0) {
//         newWeights[index] = 0.0;
//     }
//     else if (newWeights[index] > 1.0) {
//         newWeights[index] = 1.0;
//     }

//     return newWeights;
// }