#pragma once

#include "MAI.hh"

MAI_Layers ComputeActivations()
{
    for (MAI_Neuron *neuron : this->layers[0]->neurons)
            neuron->activation = neuron->weight * neuron->activation + neuron->bias;

    for (size_t i = 1; i < this->layers.size(); i++)
    {
        MAI_Layer *layer = this->layers[i];

        for (MAI_Neuron *neuron : layer->neurons)
        {
            neuron->activation = 0.0;
            for (MAI_Neuron *connectedNeuron : neuron->connections)
                neuron->activation += connectedNeuron->activation * neuron->weight;

            neuron->activation += neuron->bias;

            MAI_ActivateNeuron(neuron, layer->activation);
        }
    }
        
    return this->layers;
}