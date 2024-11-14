# Create build directory if it doesn't exist
mkdir -p build

# Navigate to build directory
cd ./build

cmake ..

make

cp onnx_squeezenet_inference ../onnx_squeezenet_inference

cd ..

# Return to project folder
./onnx_squeezenet_inference --use_cpu ./models/squeezenet1_1_Opset18.onnx ./images/fox.jpg ./synset.txt