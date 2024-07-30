#include <iostream>

#include <bitset>

#include "../headers/MAI.hh"

int main()
{
    MAI MythAI;

    MAI_Layer *inputLayer  = MythAI.CreateLayer(1, SIGMOID);
    MAI_Layer *outputLayer = MythAI.CreateLayer(1, SIGMOID);

    MythAI.layers.push_back(inputLayer);
    MythAI.layers.push_back(MythAI.CreateLayer(16, SIGMOID));
    MythAI.layers.push_back(outputLayer);

    std::vector<int> dataset;

    for (int i = 0; i <= 3; i++)
        dataset.push_back(i % 2);

    std::cout << "Dataset size: " << dataset.size() << std::endl;

    int64_t step = 0;
    int64_t steps = 100000;

    for (MAI_Neuron *neuron : inputLayer->neurons)
        std::cout << neuron->activation << " : " << neuron->weight << ", " << neuron->bias << std::endl;

    float cost = 0.0;
    
    while (true)
    {
        step += 1;

        if (step > steps)
            break;

        if (step == steps / 4)
            std::cout << "%25 TRAINED" << std::endl;
        else if (step == steps / 2)
            std::cout << "%50 TRAINED" << std::endl;
        else if (step == steps - (steps / 4))
            std::cout << "%75 TRAINED" << std::endl;


        for (int i = 0; i < dataset.size(); i++)
        {
            std::string iBinary = std::bitset<1> (i).to_string();

            for (size_t s = 0; s < iBinary.size(); s++)
                inputLayer->neurons[s]->activation = ((iBinary[s] == '1') ? 1 : 0);

            MythAI.ComputeActivations();

            MAI_Layer *desiredOutput = MythAI.CreateLayer(2, 0);
            desiredOutput->neurons[0]->activation = i%2;

            cost = MythAI.Backpropagate(desiredOutput);

            //std::cout << "Cost: " << cost << std::endl;

            MythAI.ClearInputs();
            delete desiredOutput;
        }
    }

    std::cout << "------------------" << std::endl;

    for (int i = 0; i < dataset.size(); i++)
    {
        std::string iBinary = std::bitset<1> (i).to_string();

        for (size_t s = 0; s < iBinary.size(); s++)
            inputLayer->neurons[s]->activation = ((iBinary[s] == '1') ? 1 : 0);

        MythAI.ComputeActivations();

        //std::cout << iBinary << " ------------------" << std::endl;
        //for (MAI_Neuron *neuron : inputLayer->neurons)
            //std::cout << neuron->activation << " : " << neuron->weight << ", " << neuron->bias << std::endl;

        std::cout << ":: " << i << " : " << MythAI.layers[MythAI.layers.size() - 1]->neurons[0]->activation << std::endl;

        MythAI.ClearInputs();
    }


    std::cout << "------------------" << std::endl;
    for (MAI_Neuron *neuron : inputLayer->neurons)
        std::cout << neuron->activation << " : " << neuron->weight << ", " << neuron->bias << std::endl;
    std::cout << "------------------" << std::endl;
    std::cout << std::endl << "Even: " << 2 % 2 << ", Odd: " << 1 % 2 << std::endl;

    while (true)
    {
        std::cout << "------------------" << std::endl;

        int input;

        std::cin >> input;

        std::string iBinary = std::bitset<1> (input).to_string();

        for (size_t s = 0; s < iBinary.size(); s++)
            inputLayer->neurons[s]->activation = ((iBinary[s] == '1') ? 1 : 0);

        MythAI.ComputeActivations();

        std::cout << ":: " << input << " : " << MythAI.layers[MythAI.layers.size() - 1]->neurons[0]->activation << std::endl;
    }

    return 0;
}