#ifndef BL_CLASSIFIER_H
#define BL_CLASSIFIER_H

#include "Classifier.h"

#include <tuple>

using namespace std;

class BLClassifier : public Classifier{
    protected:
        static constexpr int MAX_EVALUATIONS = 15000;
        static constexpr double MEDIA = 0.0;
        static constexpr double VARIANZA = 0.3;

        int MAX_NEIGHBORS;

    public:
        BLClassifier(const vector<vector<DataInstance>>& datos);

        tuple<vector<double>, double> train(int fold) override;

        vector<double> busquedaLocal() const;

        vector<double> generaSolucionInicial() const;

        double funcionObjetivoBL(const vector<double>& weights) const;

        double calculateClassRateInTrain(const vector<double>& weights) const;

        vector<double> generaVecino(const vector<double>& weights, int index) const;
};

#endif
