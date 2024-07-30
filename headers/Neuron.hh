#pragma once

#include <vector>
#include <random>

class MAI_Neuron
{
private:
    std::random_device device;
    std::uniform_real_distribution<double> dist;
public:
    std::vector<MAI_Neuron*> connections;

    double weight;
    double bias;
    double activation;
    double input;

    double derivativeWeight;
    double derivativeBias;

    MAI_Neuron()
    {
        this->weight = dist(device);
        this->bias = dist(device);
        this->activation = 0;
        this->input = 0;

        this->derivativeWeight = 0;
        this->derivativeBias = 0;
    }
};