#pragma once

#include <vector>

#include "Neuron.hh"

class MAI_Layer
{
public:
    std::vector<MAI_Neuron*> neurons;

    int activation;
} typedef MAI_Layer;