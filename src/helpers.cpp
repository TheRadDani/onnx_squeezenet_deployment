#include "helpers.hpp"


// Function to validate the input image file extension.
bool imageFileExtension(std::string str)
{
  // is empty throw error
  if (str.empty())
    throw std::runtime_error("[ ERROR ] The image File path is empty");

  size_t pos = str.rfind('.');
  if (pos == std::string::npos)
    return false;

  std::string ext = str.substr(pos+1);

  if (ext == "jpg" || ext == "jpeg" || ext == "gif" || ext == "png" || ext == "jfif" || 
        ext == "JPG" || ext == "JPEG" || ext == "GIF" || ext == "PNG" || ext == "JFIF") {
            return true;
  }

  return false;
}

// Function to validate the input image file extension.
// Function to read the labels from the labelFilepath.
std::vector<std::string> readLabels(std::string& labelFilepath)
{
    std::vector<std::string> labels;
    std::string line;
    std::ifstream fp(labelFilepath);
    while (std::getline(fp, line))
    {
        labels.push_back(line);
    }
    return labels;
}

// Function to validate the input model file extension.
bool checkModelExtension(const std::string& filename)
{
    if(filename.empty())
    {
        throw std::runtime_error("[ ERROR ] The Model file path is empty");
    }
    size_t pos = filename.rfind('.');
    if (pos == std::string::npos)
        return false;
    std::string ext = filename.substr(pos+1);
    if (ext == "onnx")
        return true;
    return false;
}

// Function to validate the Label file extension.
bool checkLabelFileExtension(const std::string& filename)
{
    size_t pos = filename.rfind('.');
    if (filename.empty())
    {
        throw std::runtime_error("[ ERROR ] The Label file path is empty");
    }
    if (pos == std::string::npos)
        return false;
    std::string ext = filename.substr(pos+1);
    if (ext == "txt") {
        return true;
    } else {
        return false;
    }
}

//Handling divide by zero
float division(float num, float den){
   if (den == 0) {
      throw std::runtime_error("[ ERROR ] Math error: Attempted to divide by Zero\n");
   }
   return (num / den);
}

void printHelp() {
    std::cout << "To run the model, use the following command:\n";
    std::cout << "Example: ./run_squeezenet --use_openvino <path_to_the_model> <path_to_the_image> <path_to_the_classes_file>" << std::endl;
    std::cout << "\n To Run using OpenVINO EP.\nExample: ./run_squeezenet --use_openvino squeezenet1.1-7.onnx demo.jpeg synset.txt \n" << std::endl;
    std::cout << "\n To Run on Default CPU.\n Example: ./run_squeezenet --use_cpu squeezenet1.1-7.onnx demo.jpeg synset.txt \n" << std::endl;
}