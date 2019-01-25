#include <torch/script.h> // One-stop header.
#include <stdio.h>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <memory>

std::shared_ptr<torch::jit::script::Module> MODULE;

void get_class_index(cv::Mat inputImage)
{
  std::vector<int64_t> sizes = {1, 1, inputImage.rows, inputImage.cols};
  at::TensorOptions options(at::ScalarType::Byte);
  at::Tensor tensor_image = torch::from_blob(inputImage.data, at::IntList(sizes), options);
  tensor_image = tensor_image.toType(at::kFloat);

  std::vector<torch::jit::IValue> inputs;
  inputs.emplace_back(tensor_image.cuda());
  at::Tensor result = MODULE->forward(inputs).toTensor();


  at::Tensor max_index = result.argmax();

  std::cout << max_index << std::endl;
}


int main(int argc, const char* argv[]) {


  // Loading the Nueral Net Model
  MODULE = torch::jit::load("../model.pt");

  assert(MODULE != nullptr);
  std::cout << "Model loaded. Ready to Go!\n";

  // Example OpenCV input
  cv::Mat image = cv::imread("test.png");
  get_class_index(image);


}
