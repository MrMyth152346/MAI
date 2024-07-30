#pragma once

#include "Activation.hh"
#include "Layer.hh"

#define MAI_RATE 1e-3

constexpr float MAI_Cost(float predicted, float actual) 
{ 
    return 1.0f / 2.0f * pow(predicted - actual, 2);
}

typedef std::vector<MAI_Layer*> MAI_Layers;

class MAI
{
public:
    MAI_Layers layers;

    double Backpropagate(MAI_Layer *desiredOutput)
    {
        MAI_Layer *output = this->layers[this->layers.size() - 1];

        double totalCost = 0;

        // GO THROUGH EACH OUTPUT NEURON //
        for (size_t i = 0; i < output->neurons.size(); i++)
        {
            double cost = MAI_Cost(output->neurons[i]->activation, desiredOutput->neurons[i]->activation);
            double costDerivative = (MAI_Cost(output->neurons[i]->activation + MAI_EPSILON, desiredOutput->neurons[i]->activation) - cost) / MAI_EPSILON;
            //double costDerivative = 2.0f * (float)(desiredOutput->neurons[i]->activation - output->neurons[i]->activation);

            totalCost = cost;

            // LOOP STARTING FROM THE LAST LAYER TO FIRST LAYER //
            for (int l = this->layers.size() - 1; l >= 0; l--)
            {
                for (MAI_Neuron *neuron : this->layers[l]->neurons)
                {

                    double derivativeWeight = 0.0;
                    double derivativeBias = 0.0;

                    
                    for (int l_z = l; l_z >= 0; l_z--)
                    {
                        double activationZ = 0.0;

                        if (l_z > 0)
                            for (MAI_Neuron *nextLayerNeuron : this->layers[l - 1]->neurons)
                                activationZ += nextLayerNeuron->activation;
                        else
                            activationZ += neuron->input;

                        double gradiant = costDerivative / neuron->weight;

                        switch (this->layers[l]->activation)
                        {
                            case RELU:
                            {
                                gradiant *=  MAI_RELU_DERIVATIVE(neuron->input);
                                derivativeWeight += gradiant * activationZ;
                                derivativeBias += gradiant;
                                break;
                            }
                            case SIGMOID:
                            {
                                gradiant *= MAI_SIGMOID_DERIVATIVE(neuron->input);
                                derivativeWeight += gradiant * activationZ;
                                derivativeBias += gradiant;
                                break;
                            }
                        }
                    }

                    neuron->derivativeWeight += derivativeWeight;
                    neuron->derivativeBias += derivativeBias;
                }
            }
        }

        // UPDATE WEIGHTS AND BIASES //
        for (MAI_Layer *layer : this->layers)
        {
            for (MAI_Neuron *neuron : layer->neurons)
            {
                neuron->weight -= neuron->derivativeWeight * MAI_RATE;
                neuron->bias -= neuron->derivativeBias * MAI_RATE;

                neuron->derivativeWeight = 0.0;
                neuron->derivativeBias = 0.0;
            }
        }

        return totalCost;
    }

    // ACTIVATIONS //
    void ComputeActivations()
    {
        for (MAI_Neuron *neuron : this->layers[0]->neurons)
        {
            neuron->input = neuron->activation;
            neuron->activation = neuron->weight * neuron->activation + neuron->bias;
            MAI_ActivateNeuron(neuron, this->layers[0]->activation);
        }
        
        for (size_t i = 1; i < this->layers.size(); i++)
        {
            MAI_Layer *layer = this->layers[i];

            for (MAI_Neuron *neuron : layer->neurons)
            {
                neuron->activation = 0.0;
                neuron->input = 0.0;

                for (MAI_Neuron *connectedNeuron : this->layers[i - 1]->neurons)
                    neuron->activation += connectedNeuron->activation * connectedNeuron->weight;

                neuron->activation += neuron->bias;
                neuron->input += neuron->activation;

                MAI_ActivateNeuron(neuron, layer->activation);

                //std::cout << i << " :: " << neuron->activation << '\n';
            }
        }
    }

    // LAYER //
    MAI_Layer* CreateLayer(int neurons, int activation)
    {
        MAI_Layer *layer = new MAI_Layer;

        for (int i = 0; i < neurons; i++)
        {
            MAI_Neuron *neuron = new MAI_Neuron;

            layer->neurons.push_back(neuron);
        }
        
        layer->activation = activation;

        return layer;
    }


    // INPUTTING //
    void Input(double input)
    {
        this->layers[0]->neurons[input]->activation = 1; //* this->layers[0]->neurons[0]->weight + this->layers[0]->neurons[0]->bias;

        MAI_ActivateNeuron(this->layers[0]->neurons[0], this->layers[0]->activation);
    }

    void ClearInputs()
    {
        for (MAI_Neuron *neuron : this->layers[0]->neurons)
            neuron->activation = 0;
    }

} typedef MAI;