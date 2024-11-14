#include <vector>
#include <string>
#include <exception>
#include <iostream>
#include <stdexcept> // To use runtime_error
#include <exception>
#include <fstream>

#ifndef HELPERS_HPP
#define HELPERS_HPP

bool imageFileExtension(std::string str);

// Function to read the labels from the labelFilepath.
std::vector<std::string> readLabels(std::string& labelFilepath);

// Function to validate the input model file extension.
bool checkModelExtension(const std::string& filename);

// Function to validate the Label file extension.
bool checkLabelFileExtension(const std::string& filename);

//Handling divide by zero
float division(float num, float den);

void printHelp();

#endif // HELPERS_HPP