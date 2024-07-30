#pragma once

#include <math.h>

#include "Neuron.hh"

#define MAI_EPSILON 1e-3

#define SIGMOID 0
#define RELU 1
#define HTAN 2

#define MAI_SIGMOID(x) (1.0 / (exp(-(x)) + 1.0)) 
#define MAI_HTAN(x) (exp((x)) - exp(-(x))) / (exp((x)) + exp(-(x)))
#define MAI_RELU(x) (((x) > 0.0) ? (x) : 0.0)

#define MAI_SIGMOID_DERIVATIVE(x) ((MAI_SIGMOID(x + MAI_EPSILON) - MAI_SIGMOID(x)) / MAI_EPSILON) // (1.0 / (1.0 + exp(-(x))) * (1.0 - (1.0 / (1.0 + exp(-(x))))))
#define MAI_RELU_DERIVATIVE(x) (((x) >= 0.0) ? 1.0 : 0.0)

#define MAI_SQUARED(x) (x * x)

void MAI_ActivateNeuron(MAI_Neuron *neuron, int activation)
{
    switch (activation)
    {
        
        case SIGMOID:
            neuron->activation = MAI_SIGMOID(neuron->activation);
            break;
        case RELU:
            neuron->activation = MAI_RELU(neuron->activation);
            break;
        case HTAN:
            neuron->activation = MAI_HTAN(neuron->activation);
            break;
    }
}